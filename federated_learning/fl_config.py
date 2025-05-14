# fl_config.py
import json
from dataclasses import dataclass
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_learning.fl_core import FederatedLearning
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces.output import Output

@dataclass
class AlgorithmOptions:
    key: str
    node_processing_time: int
    search_depth: int

class Config:

    def __init__(self, federated_learning: FederatedLearning, algorithm=None, debug=False):
        self.fl_core = federated_learning
        self.model = Model()
        self.options = None
        self.model_options = None
        self.algorithm = algorithm
        self.debug = debug

    def read_options(self, options):
        self.options = options
        self.set_federated_learning_model()
        self.set_federated_learning()
        self.print_json_options()

        if self.algorithm:
            self.fl_core.set_flam(self.algorithm)
            if self.debug:
                print("FLAM algorithm data passed to core:", self.algorithm)

    def read_options_from_file(file):
        return super().read_options_from_file()

    def set_federated_learning(self) -> None:
        self.fl_core.set_num_rounds(self.options["num_rounds"])
        self.fl_core.set_num_clients(self.options["num_clients"])

    def set_federated_learning_model(self) -> None:
        self.model.set_model_type(self.options["model_type"])
        self.model.set_data_set(self.options["data_set"])
        self.fl_core.set_model(self.model)

    def print_json_options(self) -> None:
        print("----Printing JSON options----")
        print(f"Number of rounds: ({self.options['num_rounds']})")
        print(f"Number of clients: ({self.options['num_clients']})")
        print(f"Model type: ({self.options['model_type']})")
        print(f"Dataset: ({self.options['data_set']})")
        print("----JSON options printed successfully----")


if __name__ == "__main__":
    fl_instance = FederatedLearning()
    config = Config(federated_learning=fl_instance, debug=False)

    test_options = {
        "num_rounds": 5,
        "num_clients": 3,
        "model_type": "SimpleCNN",
        "data_set": "MNIST"
    }

    config.read_options(test_options)