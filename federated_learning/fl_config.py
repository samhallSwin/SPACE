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

    def __init__(self, federated_learning: FederatedLearning):
        self.fl_core = federated_learning
        self.model = Model()
        self.options = None
        self.model_options = None

    def read_options(self, options: dict):
        self.options = options
        self.set_federated_learning_model()
        self.set_federated_learning()

    def read_options_from_file(self, file_path: str):
        with open(file_path, 'r') as file:
            full_config = json.load(file)
        options = full_config.get("federated_learning", {})
        self.read_options(options)

    def set_federated_learning(self) -> None:
        self.fl_core.set_num_rounds(self.options["num_rounds"])
        self.fl_core.set_num_clients(self.options["num_clients"])

        self.fl_core.print_config_summary(
        model_type=self.options["model_type"],
        data_set=self.options["data_set"]
        )
        print ("FL setters called")

    def set_federated_learning_model(self) -> None:
        self.model.set_model_type(self.options["model_type"])
        self.model.set_data_set(self.options["data_set"])
        self.fl_core.set_model(self.model)
        print ("MODEL setters called")

if __name__ == "__main__":
    from federated_learning.fl_core import FederatedLearning

    fl_instance = FederatedLearning()
    config = Config(fl_instance)

    config.read_options_from_file("options.json")