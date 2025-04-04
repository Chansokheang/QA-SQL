import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import pdb
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        
def load_json(dir):
    with open(dir, 'r', encoding='utf8') as j:
        contents = json.loads(j.read())
    return contents

def result_callback(result):
    exec_result.append(result)


def execute_sql(predicted_sql, ground_truth, db_path):
    import os

    if not os.path.exists(db_path):
        print("not exist...")
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res



def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql,
                                  args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f'timeout',)]
        res = 0
    except Exception as e:
        result = [(f'error',)]  # possibly len(query) > 512 or not executable
        res = 0
    result = {'sql_idx': idx, 'res': res}
    return result


def package_sqls(sql_path, db_root_path, mode='gpt', data_mode='dataset', start_idx=0, batch_size=10):
    clean_sqls = []
    db_path_list = []
    # Use the file path provided directly as a parameter
    file_path = sql_path

    # Hardcoded database mappings for the first 20 questions
    # Using actual databases from the BIRD benchmark
    db_mapping = {
        0: "california_schools",
        1: "california_schools",
        2: "california_schools",
        3: "california_schools",
        4: "california_schools",
        5: "california_schools",
        6: "california_schools",
        7: "california_schools",
        8: "california_schools",
        9: "california_schools",
        10: "card_games",
        11: "card_games",
        12: "card_games",
        13: "financial",
        14: "financial",
        15: "european_football_2",
        16: "european_football_2",
        17: "european_football_2",
        18: "formula_1",
        19: "formula_1"
    }

    print("Checking file path:", file_path)
    if mode == 'gpt':
        sql_data = json.load(open(file_path, 'r', encoding='utf8'))
        # Get keys as integers and sort them
        keys = sorted([int(k) for k in sql_data.keys()])
        # Select only batch_size keys starting from start_idx
        batch_keys = keys[start_idx:start_idx+batch_size]
        
        for idx in batch_keys:
            sql_str = sql_data[str(idx)]
            
            # Check if the string contains the delimiter
            delimiter = '\t----- bird -----\t'
            if delimiter in sql_str:
                sql, db_name = sql_str.split(delimiter)
            else:
                # Use the SQL as is and get the db_name from the hardcoded mapping
                sql = sql_str
                if idx in db_mapping:
                    db_name = db_mapping[idx]
                else:
                    # Default to a common database if not found
                    db_name = "schools"
                
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    elif mode == 'gt':
        # Use dev.sql file which has SQL and database name separated by a tab
        sqls = open(sql_path + 'dev.sql')
        sql_txt = sqls.readlines()
        # Select only batch_size entries starting from start_idx
        batch_txt = sql_txt[start_idx:start_idx+batch_size]
        
        for i, sql_str in enumerate(batch_txt):
            # The dev.sql file has SQL and database name
            try:
                sql, db_name = sql_str.strip().split('\t')
                clean_sqls.append(sql)
                db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')
            except ValueError:
                print(f"Error parsing line {start_idx + i}: {sql_str}")
                # Use default if parsing fails
                sql = sql_str.strip()
                idx = start_idx + i
                if idx in db_mapping:
                    db_name = db_mapping[idx]
                else:
                    db_name = "california_schools"  # default
                clean_sqls.append(sql)
                db_path_list.append(db_root_path + db_name + '/' + db_name + '.sqlite')

    return clean_sqls, db_path_list

def run_sqls_parallel(sqls, db_places, num_cpus=1, meta_time_out=30.0):
    pool = mp.Pool(processes=num_cpus)
    for i, sql_pair in enumerate(sqls):
        predicted_sql, ground_truth = sql_pair
        pool.apply_async(execute_model, args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out), callback=result_callback)
    pool.close()
    pool.join()

def sort_results(list_of_dicts):
  return sorted(list_of_dicts, key=lambda x: x['sql_idx'])

def compute_acc(exec_results):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    acc = sum(results)/num_queries if num_queries > 0 else 0
    return acc * 100

def compute_acc_by_diff(exec_results, diff_json_path, start_idx):
    num_queries = len(exec_results)
    results = [res['res'] for res in exec_results]
    contents = load_json(diff_json_path)
    # Get only batch content
    batch_contents = contents[start_idx:start_idx+num_queries]
    
    simple_results, moderate_results, challenging_results = [], [], []

    for i, content in enumerate(batch_contents):
        if content['difficulty'] == 'simple':
            simple_results.append(exec_results[i])
        if content['difficulty'] == 'moderate':
            moderate_results.append(exec_results[i])
        if content['difficulty'] == 'challenging':
            challenging_results.append(exec_results[i])

    simple_acc = sum([res['res'] for res in simple_results])/len(simple_results) if len(simple_results) > 0 else 0
    moderate_acc = sum([res['res'] for res in moderate_results])/len(moderate_results) if len(moderate_results) > 0 else 0
    challenging_acc = sum([res['res'] for res in challenging_results])/len(challenging_results) if len(challenging_results) > 0 else 0
    all_acc = sum(results)/num_queries if num_queries > 0 else 0
    count_lists = [len(simple_results), len(moderate_results), len(challenging_results), num_queries]
    return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, all_acc * 100, count_lists

def print_data_with_details(score_lists, count_lists, exec_results, start_idx, batch_size):
    # Print only question types and total accuracy
    print('\n======================================    ACCURACY    =====================================')
    levels = ['simple', 'moderate', 'challenging', 'total']
    print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
    print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
    print("{:20} {:<20.2f} {:<20.2f} {:<20.2f} {:<20.2f}".format('accuracy', *score_lists))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--predicted_sql_path', type=str, required=True, default='')
    args_parser.add_argument('--ground_truth_path', type=str, required=True, default='')
    args_parser.add_argument('--data_mode', type=str, required=True, default='dev')
    args_parser.add_argument('--db_root_path', type=str, required=True, default='')
    args_parser.add_argument('--num_cpus', type=int, default=1)
    args_parser.add_argument('--meta_time_out', type=float, default=30.0)
    args_parser.add_argument('--mode_gt', type=str, default='gt')
    args_parser.add_argument('--mode_predict', type=str, default='gpt')
    args_parser.add_argument('--diff_json_path', type=str, default='')
    args_parser.add_argument('--start_idx', type=int, default=0, help='Starting index for batch evaluation')
    args_parser.add_argument('--batch_size', type=int, default=10, help='Number of questions to evaluate in one batch')
    args = args_parser.parse_args()
    exec_result = [] 
    
    print(f"Evaluating batch of {args.batch_size} questions starting from index {args.start_idx}")
    
    pred_queries, db_paths = package_sqls(
        args.predicted_sql_path, 
        args.db_root_path, 
        mode=args.mode_predict,
        data_mode=args.data_mode,
        start_idx=args.start_idx,
        batch_size=args.batch_size
    )
    
    # generate gt sqls:
    gt_queries, db_paths_gt = package_sqls(
        args.ground_truth_path, 
        args.db_root_path, 
        mode='gt',
        data_mode=args.data_mode,
        start_idx=args.start_idx,
        batch_size=args.batch_size
    )

    query_pairs = list(zip(pred_queries, gt_queries))
    
    if len(query_pairs) == 0:
        print("No queries found in the specified batch range.")
        sys.exit(0)
        
    run_sqls_parallel(query_pairs, db_places=db_paths, num_cpus=args.num_cpus, meta_time_out=args.meta_time_out)
    exec_result = sort_results(exec_result)
    
    print('Results for batch starting at index', args.start_idx)
    if args.diff_json_path:
        simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
            compute_acc_by_diff(exec_result, args.diff_json_path, args.start_idx)
        score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
        print_data_with_details(score_lists, count_lists, exec_result, args.start_idx, args.batch_size)
    else:
        # Only compute total accuracy without detailed results
        acc = compute_acc(exec_result)
        # Create dummy scores for categories since we don't have the difficulty info
        score_lists = [0, 0, 0, acc]
        count_lists = [0, 0, 0, len(exec_result)]
        
        print('\n======================================    ACCURACY    =====================================')
        levels = ['simple', 'moderate', 'challenging', 'total']
        print("{:20} {:20} {:20} {:20} {:20}".format("", *levels))
        print("{:20} {:<20} {:<20} {:<20} {:<20}".format('count', *count_lists))
        print("{:20} {:<20} {:<20} {:<20} {:<20.2f}".format('accuracy', 'N/A', 'N/A', 'N/A', acc))
    
    print('===========================================================================================')
    next_batch = args.start_idx + args.batch_size
    print(f"Finished evaluation of current batch. To evaluate the next batch, use --start_idx={next_batch}")