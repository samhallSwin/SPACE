# FLAM Generator - Team Guide

## 🎯 Purpose
The FLAM Generator creates **Federated Learning Adjacency Matrices (FLAMs)** that define how satellites communicate during federated learning training rounds.

## 🔄 System Integration Overview

```
📡 TLE Files → 🛰️ SatSim → 🧮 Algorithm → 📊 FLAM CSV → 🤖 FL Core
```

### Current Status: ✅ **WORKING & INTEGRATED**
- **SatSim Integration**: ✅ Working with FLOMPS
- **Algorithm Integration**: ✅ Sam's enhanced algorithm integrated
- **FL Core Compatibility**: ✅ CSV format ready
- **Output**: ✅ 100-timestep files with correct round-based training

---

## 🛠️ How It Works

### **Step 1: Satellite Simulation (SatSim)** 📡
```bash
Input: TLE file (satellite orbital data)
Process: Calculate satellite positions every minute for 100 minutes
Output: Adjacency matrices (who can communicate with whom)
```

**Example**: 4 satellites from `TLEs/SatCount4.tle`
- Simulates real orbital mechanics
- 10km communication range threshold
- Generates 100 timesteps (1-minute intervals)

### **Step 2: Enhanced Algorithm** 🧮
```bash
Input: Adjacency matrices from SatSim
Process: Apply Sam's round-based federated learning logic
Output: Training phases and target assignments
```

**Algorithm Features**:
- **25 rounds** of federated learning
- **Phase Logic**:
  - Timesteps 1-3: `TRAINING` (local model updates)
  - Timestep 4+: `TRANSMITTING` (send to aggregator)
- **Dynamic Target**: Each round randomly selects aggregator satellite
- **Connection Evolution**: Links can change with 10% probability

### **Step 3: Dual Output** 📁
1. **Original Format**: `flomps_algorithm/output/flam_*.txt` (backward compatibility)
2. **Sam's CSV Format**: `synth_FLAMs/flam_*.csv` (FL core ready)

---

## 🚀 Usage for Your Team

### **Basic Generation**
```bash
# into the FLAM Generator directory
cd /path/to/your/SPACE_FLTeam
python generate_flam_csv.py
```

### **Custom Satellite Count**
```bash
# use different TLE files for different satellite counts
python generate_flam_csv.py TLEs/SatCount8.tle    # 8 satellites
python generate_flam_csv.py TLEs/SatCount40.tle   # 40 satellites
```

### **Integration with Main FLOMPS**
```bash
# Full FLOMPS run with FLAMs
cd /path/to/your/SPACE_FLTeam 
python main.py flomps --start-time "2024-09-12 12:00:00" --end-time "2024-09-12 13:40:00" 
```

---

## 📊 Output Format Example

```csv
Timestep: 1, Round: 1, Target Node: 3, Phase: TRAINING
0,0,0,0  ← No communication during training
0,0,0,0
0,0,0,0
0,0,0,0

Timestep: 4, Round: 1, Target Node: 3, Phase: TRANSMITTING
0,1,1,0  ← Satellites communicate to target (node 3)
1,0,0,1
1,0,0,0
0,1,0,0
```

**Key Fields**:
- **Timestep**: Current minute (1-100)
- **Round**: FL training round (1-25)
- **Target Node**: Which satellite aggregates models
- **Phase**: TRAINING or TRANSMITTING
- **Matrix**: 4x4 adjacency matrix (1=can communicate, 0=cannot)

---

## ⚙️ Configuration

### **Edit `options.json`** to modify:
```json
{
  "sat_sim": {
    "start_time": "2024-09-12 12:00:00",
    "end_time": "2024-09-12 13:40:00",
    "timestep": 1,  // 1-minute intervals = 100 timesteps
    "gui": false
  }
}
```

### **Available TLE Files**:
- `TLEs/SatCount4.tle` - 4 satellites (default)
- `TLEs/SatCount8.tle` - 8 satellites
- `TLEs/SatCount40.tle` - 40 satellites
- `TLEs/Walker.tle` - Walker constellation

---

## 🔧 Team Responsibilities

### **For System Integration Team**:
- **Monitor**: Ensure `generate_flam_csv.py` runs without errors
- **Verify**: Check output files are generated in `synth_FLAMs/`
- **Test**: Run with different satellite counts for scalability

### **For FL Team**:
- **Input**: Use CSV files from `synth_FLAMs/` folder
- **Validate**: Ensure FL core can parse the CSV format
- **Report**: Any compatibility issues with round-based training

### **For SatSim Team**:
- **TLE Files**: Ensure valid orbital data in `TLEs/` folder
- **Simulation**: Verify realistic communication ranges
- **Performance**: Monitor simulation time for large satellite counts

### **For Algorithm Team**:
- **Parameters**: Adjust Sam's algorithm settings if needed:
  ```python
  toggle_chance=0.1,    # 10% link change probability
  training_time=3,      # 3 timesteps of training per round
  down_bias=2.0        # Connection evolution bias
  ```

---

## 🐛 Troubleshooting

### **Common Issues**:

1. **No CSV output generated**:
   ```bash
   # Check if SatSim completed successfully
   ls -la synth_FLAMs/
   ```

2. **Wrong number of timesteps**:
   ```bash
   # Verify options.json configuration
   cat options.json | grep -A5 "sat_sim"
   ```

3. **TLE file not found**:
   ```bash
   # List available TLE files
   ls -la TLEs/
   ```

### **Success Indicators**:
- ✅ "✅ SatSim completed: 100 timesteps generated"
- ✅ "✅ Algorithm completed!"
- ✅ "📊 CSV contains 100 timesteps"
- ✅ File created in `synth_FLAMs/` folder

---

## 📈 Performance Metrics

**Current Benchmarks**:
- **4 satellites**: ~10 seconds generation time
- **8 satellites**: ~15 seconds generation time
- **40 satellites**: ~45 seconds generation time
- **File size**: ~15KB per 100 timesteps for 4 satellites

---

## 🔗 Dependencies

**Required for your testing**:
- Python 3.9+
- Skyfield (orbital mechanics)
- NumPy (matrix operations)
- Valid TLE files in `TLEs/` folder

**Integration points**:
- **Input**: SatSim adjacency matrices
- **Output**: FL-compatible CSV files
- **Config**: `options.json` settings
