import flwr as fl
import tensorflow as tf
from typing import Tuple, List
import numpy as np
import multiprocessing
import time

# Placeholder simple CNN model (we can modify this to the actual dataset later)
def create_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model

def load_mnist_data(test: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
    if test:
        return x_test, y_test
    else:
        return x_train, y_train

def start_server():
    """Start the Flower server."""

    def get_eval_fn(model):
        """Return an evaluation function for server-side evaluation."""
        x_test, y_test = load_mnist_data(test=True)

        def evaluate(parameters: fl.common.Parameters) -> Tuple[float, float]:
            # Convert parameters to weights (numpy uses weights and flwr needs parameters)
            weights = fl.common.parameters_to_ndarrays(parameters)
            model.set_weights(weights)
            loss, accuracy = model.evaluate(x_test, y_test)
            return loss, accuracy

        return evaluate

    # Load and compile model for server-side evaluation
    model = create_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Start Flower server for the first round
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3,
        # eval_fn=get_eval_fn(model),
    )
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=4), strategy=strategy, server_address="localhost:8080")

def start_client(client_id: int):
    """Start a Flower client."""

    class MNISTClient(fl.client.NumPyClient):
        def __init__(self, model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray):
            self.model = model
            self.x_train = x_train
            self.y_train = y_train

        def get_parameters(self, config: dict) -> List[np.ndarray]:
            # Return model weights as a list of numpy arrays
            return self.model.get_weights()

        def fit(self, parameters: List[np.ndarray], config: dict) -> Tuple[List[np.ndarray], int, dict]:
            # Convert parameters to weights
            self.model.set_weights(parameters)
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
            # Convert weights to parameters
            new_parameters = self.model.get_weights()
            return new_parameters, len(self.x_train), {}

        def evaluate(self, parameters: List[np.ndarray], config: dict) -> Tuple[float, int, dict]:
            # Convert parameters to weights
            self.model.set_weights(parameters)
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
            return loss, len(self.x_train), {"accuracy": accuracy}

    # Load model and data
    model = create_model()
    x_train, y_train = load_mnist_data()

    # Create Flower client
    client = MNISTClient(model, x_train, y_train)

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    # Start the server in a separate process
    server_process = multiprocessing.Process(target=start_server)
    server_process.start()

    # Give the server some time to start
    time.sleep(5)

    # Start the clients in separate processes
    client_processes = []
    for i in range(3):
        client_process = multiprocessing.Process(target=start_client, args=(i,))
        client_process.start()
        client_processes.append(client_process)

    # Wait for all client processes to complete
    for client_process in client_processes:
        client_process.join()

    # Stop the server process
    server_process.terminate()
    server_process.join()
