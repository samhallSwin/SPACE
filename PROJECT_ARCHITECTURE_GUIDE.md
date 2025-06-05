# SPACE project documentation
**Federated Learning Satellite Swarm Simulation Suite - Complete Data Structure and Component Analysis**
---

## Project Overview

SPACE (Satellite Federated Learning Project) is a complex simulation system designed to validate federated learning concepts in satellite swarms (FLOMPS - Federated Learning Over Moving Parameter Server).

### Core Objectives
- Simulate satellite orbital motion and communication links
- Implement federated learning algorithms based on topology changes
- Generate time-series adjacency matrix data
- Provide visualization and analysis tools

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SPACE Project                        │
├─────────────────────────────────────────────────────────┤
│  📡 input        🧮 process      📊 output              │
│                                                         │
│  TLE Files   →   SatSim      →   Algorithm   →   CSV    │
│  options.json →  FlompsAlgo  →   FL Core     →   Viz    │
│  Config      →   Interfaces  →   Results     →   GUI    │
└─────────────────────────────────────────────────────────┘
```

### Main Data Flow
```
TLE Data → Satellite Position Calculation → Adjacency Matrix → Federated Learning Algorithm → FLAM File → Visualization Results
```

---

## Project Structure

### 1. **core module directory**
```
SPACE_FLTeam/
├── main.py                    
├── module_factory.py          
├── cli_args.py               
├── options.json              
├── settings.json             
├── requirements.txt          
└── README.md                 
```

### 2. **Core Module Directory**

#### 🛰️ **sat_sim/** - Satellite Simulation Module
```
sat_sim/
├── sat_sim.py               
├── sat_sim_config.py        
├── sat_sim_handler.py       
├── sat_sim_output.py        
└── sat_sim_gui.py          
```
**Features**:
- Read TLE orbital data
- Calculate satellite positions and communication links
- Generate time-series adjacency matrices

#### 🧮 **flomps_algorithm/** - FLOMPS Algorithm Module
```
flomps_algorithm/
├── algorithm_core.py        # Core algorithm logic
├── algorithm_config.py      # Algorithm configuration
├── algorithm_handler.py     # Algorithm processor
├── algorithm_output.py      # Output formatting
└── output/                  # Algorithm output directory
```
**Features**:
- Implement Sam's federated learning algorithm
- Process adjacency matrix evolution
- Generate round and phase information

#### 🤖 **federated_learning/** - Federated Learning Module
```
federated_learning/
├── fl_core.py              # FL core algorithm
├── fl_config.py            # FL configuration
├── fl_handler.py           # FL processor
├── fl_output.py            # FL output management
├── fl_visualization.py     # Visualization tools
└── results_from_output/    # Results storage
```
**Features**:
- TensorFlow/PyTorch model training
- Client aggregation logic
- Performance metrics calculation

#### 🔗 **interfaces/** - Interface Abstraction Layer
```
interfaces/
├── config.py               # Configuration interface
├── handler.py              # Handler interface
├── output.py               # Output interface
└── federated_learning.py   # FL interface
```
**Features**:
- Define standard interfaces
- Ensure module compatibility

#### 🛠️ **utilities/** - Utilities Module
```
utilities/
└── path_manager.py         # Path manager
```
**Features**:
- Unified path management
- Cross-platform compatibility

### 3. **Data and Output Directories**

#### 📡 **TLEs/** - Orbital Data
```
TLEs/
├── SatCount4.tle           # 4 satellites configuration
├── SatCount8.tle           # 8 satellites configuration
├── SatCount40.tle          # 40 satellites configuration
└── Walker.tle              # Walker constellation configuration
```

#### 📊 **synth_FLAMs/** - Synthetic FLAM Output
```
synth_FLAMs/
└── flam_*.csv              # Generated FLAM CSV files
```

### 4. **Testing and Documentation**
```
├── test_integration.py         # Integration testing
├── test_complete_integration.py # Complete integration testing
├── test_fl_compatibility.py    # FL compatibility testing
├── unit_test_algorithm_component.py # Unit testing
├── TEAM_FLAM_GENERATOR_GUIDE.md # Team usage guide
└── PROJECT_ARCHITECTURE_GUIDE.md # Architecture documentation (this file)
```

---

## 🔄 Data Structure and Flow

### 1. **Input Data Formats**

#### TLE File Format (Satellite Orbital Data)
```
NOAA-18                 
1 28654U 05018A   24245.23456789  .00000123  00000-0  12345-4 0  9991
2 28654  99.1234 123.4567 0012345  12.3456 347.8901 14.12345678123456
```

#### Configuration File (options.json)
```json
{
  "sat_sim": {
    "start_time": "2024-09-12 12:00:00",
    "end_time": "2024-09-12 13:40:00",
    "timestep": 1,
    "gui": false
  },
  "algorithm": {
    "toggle_chance": 0.1,
    "training_time": 3,
    "down_bias": 2.0
  },
  "federated_learning": {
    "num_rounds": 25,
    "num_clients": 4,
    "model_type": "SimpleCNN"
  }
}
```

### 2. **Intermediate Data Structures**

#### Adjacency Matrix (numpy.ndarray)
```python
# 4x4 adjacency matrix representing communication links between 4 satellites
adjacency_matrix = np.array([
    [0, 1, 1, 0],  # Satellite 0 can communicate with satellites 1,2
    [1, 0, 0, 1],  # Satellite 1 can communicate with satellites 0,3
    [1, 0, 0, 1],  # Satellite 2 can communicate with satellites 0,3
    [0, 1, 1, 0]   # Satellite 3 can communicate with satellites 1,2
])
```

#### Timestamped Adjacency Matrix List
```python
timestamped_matrices = [
    ("2024-09-12 12:00:00", matrix_t1),
    ("2024-09-12 12:01:00", matrix_t2),
    ("2024-09-12 12:02:00", matrix_t3),
    # ... more timesteps
]
```

#### Algorithm Output Dictionary
```python
algorithm_output = {
    "2024-09-12 12:00:00": {
        'satellite_count': 4,
        'selected_satellite': "Satellite_1",
        'federatedlearning_adjacencymatrix': matrix,
        'aggregator_flag': True,
        'round_number': 1,
        'phase': 'TRAINING',
        'target_node': 0
    }
}
```

### 3. **Output Data Formats**

#### FLAM CSV Format (Sam's Format)
```csv
Timestep: 1, Round: 1, Target Node: 0, Phase: TRAINING
0,0,0,0
0,0,0,0
0,0,0,0
0,0,0,0

Timestep: 4, Round: 1, Target Node: 0, Phase: TRANSMITTING
0,1,1,0
1,0,0,1
1,0,0,1
0,1,1,0
```

#### Original TXT Format
```
time_stamp          satellite_count  satellite_name  aggregator_flag  matrix
2024-09-12 12:00:00       4           Satellite_1        True          [[0,1,1,0]...]
```

---

## ⚙️ Key Component Details

### 1. **SatSim Class** (`sat_sim/sat_sim.py`)
```python
class SatSim:
    def __init__(self, start_time, end_time, timestep, ...):
        self.satellites = {}  # Satellite dictionary
        self.adjacency_matrices = []  # Adjacency matrix list
    
    def set_tle_data(self, tle_data):
        # Set TLE orbital data
    
    def run_with_adj_matrix(self):
        # Run simulation, return timestamped adjacency matrix list
        return timestamped_matrices
```

### 2. **Algorithm Class** (`flomps_algorithm/algorithm_core.py`)
```python
class Algorithm:
    def __init__(self):
        self.satellite_names = []
        self.adjacency_matrices = []
        self.output = AlgorithmOutput()
    
    def set_algorithm_parameters(self, toggle_chance, training_time, down_bias):
        # Set Sam's algorithm parameters
    
    def start_algorithm_steps(self):
        # Execute main algorithm logic
        # Generate round, phase, target node information
```

### 3. **Path Manager** (`utilities/path_manager.py`)
```python
class ProjectPathManager:
    def __init__(self):
        self.project_root = self._find_project_root()
    
    @property
    def synth_flams_dir(self):
        # Return synth_FLAMs directory path
    
    def get_latest_csv_file(self):
        # Get the latest CSV file
```

---

## 🔧 Configuration System

### Configuration File Hierarchy
1. **settings.json** - Global system settings
2. **options.json** - Module-specific configuration
3. **Command line arguments** - Runtime overrides

### Configuration Priority
```
Command line arguments > options.json > Default values
```

---

## 🚀 Execution Modes

### 1. **Complete Workflow Mode**
```bash
python main.py flomps --start-time "2024-09-12 12:00:00" --end-time "2024-09-12 13:40:00"
```
**Flow**: TLE → SatSim → Algorithm → FL → Visualization

### 2. **Standalone Module Mode**
```bash
python generate_flam_csv.py TLEs/SatCount4.tle
```
**Flow**: TLE → SatSim → Algorithm → CSV output

### 3. **GUI Mode**
```bash
python SPACEGUI.py
```
**Flow**: Graphical interface → Visualization → Interactive control

### 4. **Test Mode**
```bash
python test_complete_integration.py
```
**Flow**: Simulated data → Algorithm testing → Output validation

---

## 📊 Performance Metrics

### System Capacity
- **Supported satellites**: 4-40 satellites
- **Timesteps**: 20-100 steps
- **Federated learning rounds**: 1-25 rounds
- **Matrix update frequency**: Every minute

### Runtime (Benchmark Tests)
- **4 satellites, 100 timesteps**: ~10 seconds
- **8 satellites, 100 timesteps**: ~15 seconds  
- **40 satellites, 100 timesteps**: ~45 seconds

### Output File Sizes
- **CSV files**: ~15KB (4 satellites, 100 timesteps)
- **TXT files**: ~10KB (original format)

---

## 🐛 Troubleshooting Guide

### Common Issues

#### 1. **Path Issues**
```bash
# Symptoms: FileNotFoundError, Permission Denied
# Solution: Use path_manager.py universal paths
from utilities.path_manager import get_synth_flams_dir
```

#### 2. **Dependency Version Conflicts**
```bash
# Symptoms: TensorFlow incompatible with Python 3.13
# Solution: Use Python 3.12 or wait for TensorFlow 2.17
```

#### 3. **CSV Format Errors**
```bash
# Symptoms: FL Core cannot parse CSV
# Solution: Check if CSV first line contains "Timestep:", "Round:", "Target Node:", "Phase:"
```

#### 4. **Memory Issues**
```bash
# Symptoms: Large-scale satellite simulation memory overflow
# Solution: Reduce timesteps or satellite count, use batch processing
```

---

## 🔮 Future Development Directions

### 1. **Algorithm Enhancement**
- More complex federated learning algorithms
- Adaptive topology optimization
- Multi-layer network support

### 2. **Visualization Improvements**
- Real-time 3D satellite orbit display
- Interactive algorithm parameter adjustment
- Performance metrics dashboard

### 3. **Deployment Optimization**
- Docker containerization
- Cloud distributed computing
- GPU acceleration support

### 4. **Data Formats**
- Support more satellite data formats
- Real-time data stream integration
- Standardized interface protocols

---

## 📚 Related Documentation

- `README.md` - Basic project introduction
- `TEAM_FLAM_GENERATOR_GUIDE.md` - FLAM generator usage guide
- `requirements.txt` - Dependency list
- `test_algorithm_component_report.txt` - Test reports

---

## 👥 Contributors

- **Elysia Guglielmo** - System Architect
- **Yuganya Perumal** - Algorithm Development
- **Sam** - Federated Learning Algorithm Expert

---

*This document is continuously updated. Please contact the project maintenance team if you have any questions.* 