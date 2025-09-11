# FLOMPS Algorithm Core - README

## Class Structure

### `Algorithm` Class

The main class that orchestrates the entire FLOMPS algorithm process.

#### **Initialization (`__init__`)**
```python
def __init__(self):
```
**Purpose**: Sets up the algorithm with default values and tracking variables.

**Key Attributes**:
- `satellite_names`: List of satellite identifiers
- `adjacency_matrices`: Time-series connectivity data from satellite simulation
- `selection_counts`: Load balancing tracker (how often each satellite serves)
- `round_number`: Current federated learning round
- `output_to_file`: Whether to save results to files

---

## Core Functions

### 1. **Data Management Functions**

#### `set_satellite_names(satellite_names)`
- **Purpose**: Configure satellite constellation
- **Input**: List of satellite names/identifiers
- **Action**: Initializes load balancing counters for each satellite

#### `set_adjacency_matrices(adjacency_matrices)`
- **Purpose**: Load connectivity data from satellite simulator
- **Input**: Time-series of connectivity matrices
- **Format**: `[(timestamp, connectivity_matrix), ...]`

#### `get_algorithm_output()`
- **Purpose**: Retrieve processed federated learning results
- **Returns**: Complete algorithm output data structure

---

### 2. **Server Selection Functions**

#### `find_best_server_for_round(start_matrix_index=0)`
**Purpose**: Select the optimal satellite to serve as federated learning coordinator for the next round.

**Algorithm**:
1. **Evaluate each satellite** as potential server
2. **Calculate time needed** for each to connect to ALL others
3. **Apply load balancing penalty** to frequently selected satellites
4. **Select satellite with minimum adjusted time**

**Key Features**:
- **Time-based optimization**: Minimizes round duration
- **Load balancing**: Prevents server monopolization
- **Predictive feedback**: Shows expected round performance

**Returns**: `(best_server_index, predicted_timestamps)`

---

#### `calculate_cumulative_connection_time(server_idx, start_matrix_index, max_lookahead=20)`
**Purpose**: Calculate how many timestamps a specific satellite needs to connect to ALL others.

**Cumulative Connectivity Model**:
- **Persistent Connections**: Once connected, satellites stay connected for the round
- **Progressive Building**: Accumulates connections over time
- **Realistic Modeling**: Matches actual satellite communication behavior

**Algorithm**:
1. **Initialize target set**: All satellites except the server
2. **Scan future timestamps**: Look ahead up to `max_lookahead`
3. **Accumulate connections**: Add newly connected satellites to set
4. **Check completion**: Return when all targets are connected
5. **Timeout protection**: Return penalty if never achieves full connectivity

**Returns**: Number of timestamps needed for full connectivity

---

#### `analyze_all_satellites(start_matrix_index, max_lookahead=20)`
**Purpose**: Comprehensive analysis of all satellites' connectivity potential.

**Provides**:
- **Connectivity timelines** for each satellite
- **Performance comparisons** across constellation
- **Detailed connectivity statistics**
- **Round feasibility analysis**

---

#### `analyze_single_satellite(sat_idx, start_matrix_index, max_lookahead=20)`
**Purpose**: Deep dive analysis of a specific satellite's connectivity pattern.

**Returns**:
- **Connection timeline**: When each target satellite connects
- **Performance metrics**: Speed and efficiency statistics
- **Feasibility assessment**: Whether full connectivity is achievable

---

### 3. **Main Algorithm Execution**

#### `start_algorithm_steps()`
**Purpose**: Execute the complete FLOMPS federated learning algorithm.

**Algorithm Flow**:
1. **Round Initialization**
   - Select optimal server using time-based optimization
   - Set round parameters and tracking variables

2. **Connectivity Monitoring**
   - Track cumulative connections for selected server
   - Continue until server connects to ALL satellites
   - Generate federated learning output for each timestamp

3. **Round Completion**
   - Detect when full connectivity is achieved
   - Finalize round with actual performance metrics
   - Prepare for next round

4. **Safety Mechanisms**
   - **Timeout protection**: Prevent infinite rounds (20 timestamp max)
   - **Data validation**: Handle edge cases and sparse connectivity
   - **Progress tracking**: Detailed logging and feedback

**Output Structure**:
Each timestamp generates:
- **Satellite Information**: Count, names, selected server
- **Connectivity Data**: Current adjacency matrix
- **Round Metadata**: Round number, phase, progress
- **Performance Metrics**: Connections achieved, targets remaining
- **Federated Learning Data**: Formatted for FL component consumption

---

## Key Algorithm Features

### **1. Time-Based Optimization**
- **Minimizes round duration** rather than maximizing connections
- **Predictive selection** based on future connectivity analysis
- **Efficient resource utilization** for faster federated learning

### **2. Cumulative Connectivity Model**
- **Realistic satellite behavior**: Connections persist over time
- **Progressive round building**: Satellites connect incrementally
- **Accurate performance prediction**: Better round planning

### **3. Advanced Load Balancing**
- **Selection count tracking**: Monitors satellite usage
- **Penalty-based fairness**: Discourages server monopolization
- **Distributed workload**: Ensures constellation-wide participation

### **4. Robust Error Handling**
- **Timeout protection**: Prevents infinite loops
- **Sparse connectivity handling**: Works with limited satellite visibility
- **Data validation**: Handles edge cases gracefully

### **5. Comprehensive Output**
- **Detailed round tracking**: Complete federated learning metadata
- **Performance analytics**: Connection statistics and timing
- **Multi-format export**: TXT and CSV output formats

---

## Usage Example

```python
# Initialize algorithm
algorithm = Algorithm()

# Configure satellite constellation
algorithm.set_satellite_names(['Sat1', 'Sat2', 'Sat3', 'Sat4'])

# Load connectivity data from satellite simulator
algorithm.set_adjacency_matrices(connectivity_data)

# Execute FLOMPS algorithm
algorithm.start_algorithm_steps()

# Retrieve results
results = algorithm.get_algorithm_output()
```

---

## Integration Points

### **Input**: Satellite Simulation Data
- **Source**: `sat_sim` component
- **Format**: Time-series adjacency matrices
- **Content**: Satellite connectivity patterns over time

### **Output**: Federated Learning Data
- **Destination**: `federated_learning` component
- **Format**: FLAM (Federated Learning Adjacency Matrix)
- **Content**: Optimized server selections and connectivity data

### **Configuration**: Algorithm Parameters
- **Load balancing penalty**: Adjustable fairness factor
- **Timeout limits**: Maximum round duration
- **Lookahead window**: Prediction depth for server selection

---
