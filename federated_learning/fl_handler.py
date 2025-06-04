"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round.
Author: Nicholas Paul Candra
Date: 2025-05-12
Version: 1.5
Python Version: 3.10+
"""

import sys
import os
import ast
import pandas as pd
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces.handler import Handler
from federated_learning.fl_core import FederatedLearning


class FLHandler(Handler):
    def __init__(self, fl_core: FederatedLearning):
        super().__init__()
        self.federated_learning = fl_core

    def parse_input(self, file):
        return self.parse_file(file)

    def run_module(self):
        if self.flam is not None:
            print("[INFO] Parsing FLAM metadata for simulation...")
            self.run_flam_based_simulation()
        else:
            print("[INFO] Running default Federated Learning Core...")
            self.federated_learning.run()

    def run_flam_based_simulation(self):
        print("[DEBUG] Parsed FLAM Columns:", self.flam.columns.tolist())

        for timestamp, group in self.flam.groupby("time_stamp"):
            print(f"\n[FLAM ROUND] \nTime: {timestamp}")

            matrix_row = group.iloc[0]
            matrix_str = matrix_row["federatedlearning_adjacencymatrix"]

            try:
                matrix = self.parse_adjacency_matrix(matrix_str)
                aggregator_id = int(matrix_row.get("aggregator_id", 0))  # Extracted earlier from header

                self.federated_learning.set_topology(matrix, aggregator_id)
                self.federated_learning.reset_clients()
                self.federated_learning.run()

            except Exception as e:
                print(f"[WARN] Error in FLAM round: {e}")
                continue

    def parse_adjacency_matrix(self, matrix_str):
        cleaned = (
            matrix_str.replace('\n', '')
                      .replace('\r', '')
                      .replace('\x00', '')
                      .replace('], [', '],[')
                      .strip().rstrip(',')
        )

        if not cleaned.startswith('[['):
            cleaned = f'[{cleaned}]'

        left, right = cleaned.count('['), cleaned.count(']')
        if left > right:
            cleaned += ']' * (left - right)
        elif right > left:
            cleaned = '[' * (right - left) + cleaned

        return ast.literal_eval(cleaned)


def load_flam_txt(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    flam_entries = []
    current_header = None
    matrix_lines = []

    for line in lines:
        # Detect header line
        if line.lower().startswith("timestep:"):
            if current_header and matrix_lines:
                flam_entries.append(process_flam_block(current_header, matrix_lines))
            current_header = line
            matrix_lines = []
        else:
            matrix_lines.append(line)

    # Append the last block
    if current_header and matrix_lines:
        flam_entries.append(process_flam_block(current_header, matrix_lines))

    if not flam_entries:
        raise ValueError("No valid FLAM blocks found in the file.")

    return pd.DataFrame(flam_entries)


def process_flam_block(header: str, matrix_lines: list) -> dict:
    # Parse metadata from header
    meta = {}
    for part in header.split(','):
        if ':' in part:
            key, val = part.strip().split(':', 1)
            meta[key.strip().lower().replace(" ", "_")] = val.strip()

    # Parse matrix
    matrix = [list(map(int, row.split(','))) for row in matrix_lines]

    return {
        "time_stamp": f"T{meta.get('timestep', '0')}",
        "satellite_count": len(matrix),
        "satellite_name": None,
        "aggregator_flag": True,
        "aggregator_id": meta.get('target_node', 0),
        "federatedlearning_adjacencymatrix": str(matrix)
    }


    return pd.DataFrame([flam_entry])


if __name__ == "__main__":
    print("[START] Initializing Federated Learning Handler...")

    fl_core = FederatedLearning()
    fl_core.set_num_clients(3)
    fl_core.set_num_rounds(1)

    if not hasattr(fl_core, 'reset_clients'):
        def reset_clients():
            fl_core.client_data = []
        fl_core.reset_clients = reset_clients

    handler = FLHandler(fl_core)

    # Robust file discovery: CSV, TXT, JSON
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flam_dir = os.path.join(script_dir, "..", "synth_FLAMs", "synth_FLAMs")
    supported_exts = ["*.csv", "*.txt", "*.json"]

    flam_files = []
    for ext in supported_exts:
        flam_files.extend(glob.glob(os.path.join(flam_dir, f"sim_{ext}")))

    if not flam_files:
        raise FileNotFoundError(f"No FLAM files found in {flam_dir} (supported: .csv, .txt, .json)")

    print(f"[INFO] Using FLAM file: {flam_files[0]}")
    handler.flam = load_flam_txt(flam_files[0])

    handler.run_module()
    print("\n[DONE] FL process complete.")
