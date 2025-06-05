# SPACE Project Quick Reference Guide
**Understand the entire project structure in 30 seconds**

---

## 🎯 Core Concepts
**SPACE = Satellite Swarm Federated Learning Simulation System**
- Simulate satellite orbital motion
- Calculate inter-satellite communication links
- Run federated learning algorithms
- Generate training schedule files

---

## 📁 Key Directories (Focus on these only)

```
SPACE_FLTeam/
├── 🚀 main.py              # Main program - run complete simulation
├── 📊 generate_flam_csv.py # Quick generator - generate CSV files only
├── ⚙️ options.json         # Configuration file - set parameters
├── 📡 TLEs/               # Satellite data - choose satellite count
├── 📈 synth_FLAMs/        # Output results - CSV files here
└── 🧪 test_*.py          # Test files - verify functionality
```

---

## 🔄 3-Step Data Flow

```
1️⃣ TLE Orbital Data  →  2️⃣ Satellite Simulation  →  3️⃣ Federated Learning Algorithm  →  📄 CSV Results
   (Satellite positions)    (Communication links)       (Training schedule)             (Timetable)
```

---

## 🚀 Quick Start Commands

### Test if system works properly
```bash
python test_complete_integration.py
```

### Generate FLAM files (Recommended for beginners)
```bash
python generate_flam_csv.py
```

### Run complete simulation
```bash
python main.py flomps --start-time "2024-09-12 12:00:00" --end-time "2024-09-12 13:40:00"
```

### Launch GUI interface
```bash
python SPACEGUI.py
```

---

## 📊 Understanding Output Files

### CSV File Location
```
synth_FLAMs/flam_4n_100t_flomps_2025-06-04_20-53-28.csv
```
- `4n` = 4 satellites
- `100t` = 100 timesteps
- `flomps` = using FLOMPS algorithm
- `2025-06-04_20-53-28` = generation time

### CSV Content Format
```csv
Timestep: 1, Round: 1, Target Node: 0, Phase: TRAINING
0,0,0,0    ← Training phase, no communication
0,0,0,0
0,0,0,0
0,0,0,0

Timestep: 4, Round: 1, Target Node: 0, Phase: TRANSMITTING  
0,1,1,0    ← Transmission phase, sending to target node 0
1,0,0,1
1,0,0,1
0,1,1,0
```

---

## ⚙️ Important Configuration (options.json)

### Modify simulation time
```json
{
  "sat_sim": {
    "start_time": "2024-09-12 12:00:00",  ← Start time
    "end_time": "2024-09-12 13:40:00",    ← End time
    "timestep": 1                         ← Time interval (minutes)
  }
}
```

### Modify algorithm parameters
```json
{
  "algorithm": {
    "toggle_chance": 0.1,     ← Connection change probability
    "training_time": 3,       ← Training timesteps
    "down_bias": 2.0         ← Connection evolution bias
  }
}
```

---

## 🛰️ Choose Satellite Count

### Available TLE files
```
TLEs/SatCount4.tle   ← 4 satellites (quick test)
TLEs/SatCount8.tle   ← 8 satellites (medium scale)
TLEs/SatCount40.tle  ← 40 satellites (large scale test)
```

### Usage
```bash
python generate_flam_csv.py TLEs/SatCount8.tle
```

---

## 🐛 Common Problem Solutions

### Problem 1: File not found
```bash
# Make sure running in project root directory
cd /path/to/SPACE_FLTeam
pwd  # Should show project path
```

### Problem 2: Permission errors
```bash
# Check if synth_FLAMs directory exists
ls -la synth_FLAMs/
# If doesn't exist, will be created automatically
```

### Problem 3: Empty CSV files
```bash
# Check output from last run
python test_fl_compatibility.py
```

### Problem 4: TensorFlow errors
```bash
# Currently known Python 3.13 incompatibility, core functions still work
# Core algorithm doesn't depend on TensorFlow
```

---

## 📈 Performance Reference

| Satellites | Timesteps | Runtime | File Size |
|------------|-----------|---------|-----------|
| 4          | 100       | ~10 sec | ~15KB     |
| 8          | 100       | ~15 sec | ~30KB     |
| 40         | 100       | ~45 sec | ~150KB    |

---

## 🔍 Verify Results

### Check if CSV files are correct
```bash
# View latest generated file
ls -la synth_FLAMs/ | tail -1

# Check file content
head -10 synth_FLAMs/flam_*.csv

# Run compatibility test
python test_fl_compatibility.py
```

### Success indicators
- ✅ CSV files generated in `synth_FLAMs/` directory
- ✅ Filename contains satellite count and timesteps
- ✅ CSV first line contains "Timestep:", "Round:", "Target Node:", "Phase:"
- ✅ All tests pass

---

## 🆘 Need Help?

1. **View detailed documentation**: `PROJECT_ARCHITECTURE_GUIDE.md`
2. **Team guide**: `TEAM_FLAM_GENERATOR_GUIDE.md`
3. **Run tests**: `python test_complete_integration.py`
4. **Check logs**: Console output during program execution

---
