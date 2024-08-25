import numpy as np

from interfaces.output import Output

class SatSimOutput(Output):
    """
    Prepares and sends data to Federated Learning module and handles writing to files.
    """

    def __init__(self):
        self.fl_input = None

    def write_to_file(self, file, matrices):
        """
        Writes matrices to the specified file with timestamps.
        """
        with open(file, "a") as f:
            for timestamp, matrix in matrices:
                f.write(f"Time: {timestamp}\n")
                np.savetxt(f, matrix, fmt='%d')
                f.write("\n")

    def set_result(self):
        """
        Placeholder for setting results, potentially integrating with a learning module.
        """
        pass

    def set_fl_input(self, fl_input):
        """
        Sets the input for the Federated Learning module, if applicable.
        """
        self.fl_input = fl_input
