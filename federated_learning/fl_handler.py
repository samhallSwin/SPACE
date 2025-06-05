"""
Filename: fl_handler.py
Description: Reads algorithm data to perform preprocessing before running federated learning round.
Author: Nicholas Paul Candra
Date: 2025-05-12
Version: 2.6
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
        self.current_round = 1

    def parse_input(self, file):
        return self.parse_file(file)

    def run_module(self):
        if self.flam is not None:
            print("[INFO] Parsing FLAM metadata for simulation...")
            print(self.flam.head())
            self.run_flam_based_simulation()
        else:
            print("[INFO] Running default Federated Learning Core...")
            self.federated_learning.run()

    def run_flam_based_simulation(self):
        print("[DEBUG] Parsed FLAM Columns:", self.flam.columns.tolist())
        
        # Initialize FL system only once
        if not hasattr(self, '_fl_initialized'):
            print("[INFO] Initializing FL system for first time...")
            self.federated_learning.initialize_data()
            self.federated_learning.initialize_model()
            self._fl_initialized = True

        for timestamp, group in self.flam.groupby("time_stamp"):
            matrix_row = group.iloc[0]
            matrix_raw = matrix_row["federatedlearning_adjacencymatrix"]
            phase = str(matrix_row.get("phase", "TRAINING")).strip().upper()

           
            try:
                aggregator_id = int(str(matrix_row.get("aggregator_id", 0)).strip())
            except ValueError:
                aggregator_id = 0

            try:
                if isinstance(matrix_raw, str):
                    matrix = self.parse_adjacency_matrix(matrix_raw)
                else:
                    matrix = matrix_raw

                print(f"\nTimestep: {timestamp}, Round: {self.current_round}, Target Node: {aggregator_id}, Phase: {phase}")
                for row in matrix:
                    print(",".join(map(str, row)))

                # Set topology but don't reinitialize everything
                self.federated_learning.set_topology(matrix, aggregator_id)
                
                # Run the FL round with the current phase
                self.federated_learning.run_flam_round({"phase": phase})

                if phase == "TRANSMITTING":
                    self.current_round += 1

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

    def load_flam_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.txt', '.csv']:
            return self._load_flam(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
            df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
            return df
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _load_flam(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

        flam_entries = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("Timestep:"):
                current_header = line
                matrix_lines = lines[i+1:i+6]
                entry = self.process_flam_block(current_header, matrix_lines)
                flam_entries.append(entry)
                i += 6
            else:
                i += 1

        return pd.DataFrame(flam_entries)

    def process_flam_block(self, header: str, matrix_lines: list) -> dict:
        import numpy as np

        header_parts = [h.strip() for h in header.split(',')]

        timestep = int(header_parts[0].split(':')[1])
        sat_count = int(header_parts[1].split(':')[1])
        sat_names = header_parts[2].split(':')[1].split()
        agg_flag = header_parts[3].split(':')[1].strip().lower() == 'true'

       
        aggregator_id = 0  # Default
        for part in header_parts:
            if "Target Node" in part:
                try:
                    aggregator_id = int(part.split(':')[1].strip())
                except ValueError:
                    aggregator_id = 0

        # Extract Phase (fallback to TRAINING)
        phase = 'TRAINING'
        for part in header_parts:
            if "Phase" in part:
                phase = part.split(':')[1].strip().upper()

        matrix_data = []
        for line in matrix_lines:
            if line.strip():
                cleaned_line = line.replace(",", " ").strip()
                row = list(map(int, cleaned_line.split()))
                matrix_data.append(row)

        adjacency_matrix = np.array(matrix_data)

        return {
            'time_stamp': timestep,
            'satellite_count': sat_count,
            'satellite_names': sat_names,
            'aggregator_flag': agg_flag,
            'aggregator_id': aggregator_id,  
            'federatedlearning_adjacencymatrix': adjacency_matrix,
            'phase': phase
        }


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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    flam_dir = os.path.join(script_dir, "..", "synth_FLAMs", "synth_FLAMs")
    supported_exts = ["*.csv", "*.txt", "*.json"]

    flam_files = []
    for ext in supported_exts:
        flam_files.extend(glob.glob(os.path.join(flam_dir, f"sim_{ext}")))

    if not flam_files:
        raise FileNotFoundError(f"No FLAM files found in {flam_dir} (supported: .csv, .txt, .json)")

    print(f"[INFO] Using FLAM file: {flam_files[0]}")
    handler.flam = handler.load_flam_file(flam_files[0])

    handler.run_module()
    print("\n[DONE] FL process complete.")
