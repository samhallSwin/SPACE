import numpy as np
import csv


from interfaces.output import Output


class SatSimOutput:
    # Prepares and sends data to Federated Learning module and handles writing to files.
    def __init__(self):
        self.fl_input = None
        self.matrices = None

    def write_to_file(self, file, matrices):
     
        self.matrices = matrices
        try:
            if not matrices:
                print(f"No data to write to {file}.")
                return False

            with open(file, "a") as f:
                for timestamp, matrix in matrices:
                    if matrix.size == 0:
                        print(f"Skipping empty matrix at {timestamp}")
                        continue
                    f.write(f"Time: {timestamp}\n")
                    np.savetxt(f, matrix, fmt='%d')
                    f.write("\n")
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False

    def write_to_csv(self, csv_file, matrices):
       
        self.matrices = matrices
        try:
            if not matrices:
                print(f"No data to write to {csv_file}.")
                return False

            # Get maximum matrix size to ensure all rows are extended accordingly
            max_size = max((len(matrix) for _, matrix in matrices), default=0)

            if max_size == 0:
                print(f"No valid matrix data to write to CSV {csv_file}.")
                return False

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                for timestamp, matrix in matrices:
                    if matrix.size == 0:
                        print(f"Skipping empty matrix at {timestamp}")
                        continue
                    writer.writerow(['Time:', timestamp, 'Size:', len(matrix)])
                    for row in matrix:
                        # Extend rows with zeros to ensure uniformity in column size
                        if len(row) < max_size:
                            row = np.pad(row, (0, max_size - len(row)), mode='constant')
                        writer.writerow(row)
                    writer.writerow([])  # Add an empty row for better readability

            return True
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            return False

    def set_result(self):
       
        pass

    def set_fl_input(self, fl_input):
        
        self.fl_input = fl_input

    def get_result(self):
        
        return self.matrices
