#!/usr/bin/env python3
"""
Evidence Extractor Tool

This script extracts evidence from the dev.json file and creates a persistent 
Chroma database to store the evidence vectors in the evidence directory.
"""

import os
import sys
import json
import argparse
from utils import create_persistent_db
from pathlib import Path

def create_evidence_directory(base_dir=None):
    """
    Creates the evidence directory if it doesn't exist
    
    Args:
        base_dir (str): Optional base directory path
    """
    if base_dir:
        evidence_dir = os.path.abspath(base_dir)
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        evidence_dir = os.path.join(current_dir, "evidence")
    
    if not os.path.exists(evidence_dir):
        os.makedirs(evidence_dir)
        print(f"Created evidence directory at {evidence_dir}")
    
    return evidence_dir

def main():
    parser = argparse.ArgumentParser(description="Extract evidence and create a persistent vector database")
    parser.add_argument("--dev_file", type=str, default="../database/dev_folder/dev.json", 
                        help="Path to the dev.json file")
    parser.add_argument("--output_dir", type=str, default="./evidence",
                        help="Directory to store evidence files (default: ./evidence)")
    parser.add_argument("--chroma_dir", type=str, default="../database/chroma/evidence_chroma_db",
                        help="Directory to store the Chroma database")
    
    args = parser.parse_args()
    
    # Create evidence directory if it doesn't exist
    evidence_dir = create_evidence_directory(args.output_dir)
    
    # Check if dev.json exists
    dev_path = os.path.abspath(args.dev_file)
    if not os.path.exists(dev_path):
        print(f"Error: Dev file not found at {dev_path}")
        return 1
    
    print(f"Processing dev file: {dev_path}")
    
    try:
        # Read dev.json file
        with open(dev_path, 'r') as f:
            data = json.load(f)
        
        # Group evidence by database ID
        grouped_evidence = {}
        for item in data:
            db_id = item["db_id"]
            evidence = item["evidence"]
            
            # Group evidence by db_id
            if db_id not in grouped_evidence:
                grouped_evidence[db_id] = []
            grouped_evidence[db_id].append(evidence)
        
        # Save grouped evidence to separate files
        for db_id, evidences in grouped_evidence.items():
            output_file = os.path.join(evidence_dir, f"{db_id}_evidence.json")
            with open(output_file, "w") as f:
                json.dump(evidences, f, indent=4)
            print(f"Created evidence file for {db_id} with {len(evidences)} items")
        
        # Create the persistent Chroma database
        chroma_dir = os.path.abspath(args.chroma_dir)
        if not os.path.exists(os.path.dirname(chroma_dir)):
            os.makedirs(os.path.dirname(chroma_dir))
        
        print(f"Creating persistent database at {chroma_dir}...")
        vector_store = create_persistent_db()
        if vector_store:
            print(f"Successfully created persistent database")
            return 0
        else:
            print("Failed to create persistent database")
            return 1
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())