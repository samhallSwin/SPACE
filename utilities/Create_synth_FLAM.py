#This is a tool to create quick and dirty FLAM outputs - data is randomly generated rather than based on SatSim. Follows the following broad steps:
# 1. Generates a set of connection matrices by giving a chance to toggle a connection each timestep (eg. isn't just totally random, it's evolving)
# 2. At the start of a round, picks a random Sat to be the PS.
# 3. zeros a number of Matrices as defined by training_time
# 4. Counts the number of steps it takes for all devices to connect to the PS
# 5. Returns to step 2 until all timesteps are done, then outputs as CSV

import argparse
import random
import numpy as np
import os
from collections import deque
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Create synthesized FLAM's for testing and exports as CSV. NOT BASED ON SAT SIM, JUST ORGANISED RANDOM DATA")
    parser.add_argument("Num_devices", type=int, help="Number of satellites in the network")
    parser.add_argument("timesteps", type=int, help="Total number of timesteps to simulate")
    parser.add_argument("toggle_chance", type=float, help="Base probability for toggling a connection, 0.1 or 0.2 seems about right")
    parser.add_argument("training_time", type=int, help="The number of timesteps it takes to train at the start of each round - no connections allowed for this time")
    parser.add_argument("--down_bias", type=float, default=1.0, help="Bias factor: random toggling led to rounds being too short, set this to greater than 1 (start at 2.0?) to make breaking a connection more likely than forming one")
    return parser.parse_args()

def is_reachable(matrix, target):
    """Check if all nodes can reach the target via BFS."""
    num_nodes = len(matrix)
    visited = [False] * num_nodes
    queue = deque([target])
    visited[target] = True

    while queue:
        node = queue.popleft()
        for neighbor, connected in enumerate(matrix[node]):
            if connected and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return all(visited)

def simulate_rounds(num_devices, timesteps, toggle_chance, training_time, down_bias):
    connections = np.zeros((num_devices, num_devices), dtype=int)

    round_number = 1
    target_node = random.randint(0, num_devices - 1)
    training_counter = 0
    in_training = True
    export_data = []

    for timestep in range(timesteps):
        # Toggle connections with directional bias
        for i in range(num_devices):
            for j in range(i + 1, num_devices):
                if connections[i][j] == 1:
                    if random.random() < toggle_chance * down_bias:
                        connections[i][j] = 0
                        connections[j][i] = 0
                else:
                    if random.random() < toggle_chance:
                        connections[i][j] = 1
                        connections[j][i] = 1

        # Create a matrix copy and zero if in training
        effective_matrix = np.copy(connections)
        phase = "TRAINING"
        if in_training:
            effective_matrix[:, :] = 0
            training_counter += 1
            if training_counter >= training_time:
                in_training = False
        else:
            phase = "TRANSMITTING"

        print(f"\n--- Timestep {timestep + 1} ---")
        print(f"Round: {round_number} | Target Node: {target_node} | Phase: {phase}")
        print(effective_matrix)

        # Save timestep data
        row = {
            "timestep": timestep + 1,
            "round": round_number,
            "target_node": target_node,
            "phase": phase
        }
        for i in range(num_devices):
            for j in range(num_devices):
                row[f"conn_{i}_{j}"] = effective_matrix[i][j]
        export_data.append(row)

        # Check for round completion
        if not in_training and is_reachable(effective_matrix, target_node):
            print(f"Round {round_number} complete — all nodes can reach target {target_node}.")
            round_number += 1
            target_node = random.randint(0, num_devices - 1)
            training_counter = 0
            in_training = True
            print(f" Starting Round {round_number} | New Target Node: {target_node}")

    # create a file with timestamp and params used so it doesn't overwrite
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = "synth_FLAMs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"sim_{num_devices}n_{timesteps}t_{toggle_chance:.2f}tc_{training_time}tr_{down_bias:.2f}db_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, mode='w', newline='') as f:
        for row in export_data:
            f.write(f"Timestep: {row['timestep']}, Round: {row['round']}, "
                    f"Target Node: {row['target_node']}, Phase: {row['phase']}\n")
            for i in range(num_devices):
                line = ",".join(str(row[f"conn_{i}_{j}"]) for j in range(num_devices))
                f.write(line + "\n")
            f.write("\n")

    print(f"\n Results exported to {filename}")

def main():
    args = parse_args()
    if not (0 <= args.toggle_chance <= 1):
        print("Error: toggle_chance must be between 0 and 1.")
        return
    simulate_rounds(args.Num_devices, args.timesteps, args.toggle_chance, args.training_time, args.down_bias)

if __name__ == "__main__":
    main()
