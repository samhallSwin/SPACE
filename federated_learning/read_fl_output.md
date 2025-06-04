# Federated Learning Output Module Documentation

## Overview
The `fl_output.py` module is a comprehensive tool for evaluating, logging, and analyzing Federated Learning models. It provides a robust interface for handling model evaluation metrics, saving results, and generating detailed performance reports.

## Features

### 1. Model Evaluation
- Evaluates trained models on test datasets
- Computes key metrics:
  - Accuracy
  - Loss
  - Processing time
  - Evaluation time
- Supports both custom and default (MNIST) test datasets

### 2. Metrics Collection
- Basic Metrics:
  - Model accuracy
  - Loss values
  - Processing time
  - Evaluation time
- Advanced Metrics:
  - Confusion matrix
  - Per-class metrics (precision, recall, F1-score)
- Custom metrics support through `add_metric()`

### 3. Result Logging
- Console output for immediate feedback
- File logging in multiple formats:
  - Text logs (.log)
  - JSON format (.json)
  - Model state (.pt)
- Timestamp-based file naming for result tracking

### 4. Model Persistence
- Save trained models for later use
- Load models for evaluation
- Compatible with PyTorch model format

## Usage Guide

### Basic Usage

1. **Initialization**
```python
from fl_output import FLOutput

# Initialize with default MNIST test dataset
output = FLOutput()

# Or initialize with custom test dataset
output = FLOutput(test_dataset=your_dataset, batch_size=32)
```

2. **Model Evaluation**
```python
# Evaluate a trained model
metrics = output.evaluate_model(trained_model, processing_time=10.5)
```

3. **Accessing Results**
```python
# Get all metrics
results = output.get_result()

# Get specific metrics
accuracy = results["accuracy"]
loss = results["loss"]
```

4. **Logging Results**
```python
# Log to console and file
output.log_result("results.log")

# Save metrics in JSON format
output.write_to_file("metrics.json", format="json")
```

### Advanced Usage

1. **Adding Custom Metrics**
```python
# Add custom metrics
output.add_metric("client_variance", 0.023)
output.add_metric("communication_rounds", 5)
```

2. **Computing Advanced Metrics**
```python
# Generate confusion matrix
confusion_matrix = output.compute_confusion_matrix(num_classes=10)

# Calculate per-class metrics
per_class_metrics = output.compute_per_class_metrics(num_classes=10)
```

3. **Saving Models**
```python
# Save the evaluated model
output.save_model("trained_model.pt")
```

### Output File Structure

All output files are saved in the `results_from_output` directory with timestamp-based naming:

```
results_from_output/
├── results_YYYYMMDD_HHMMSS.log    # Detailed log file
├── metrics_YYYYMMDD_HHMMSS.json   # Metrics in JSON format
└── model_YYYYMMDD_HHMMSS.pt       # Saved model state
```

### File Formats

1. **Log File (.log)**
```
--- Federated Learning Results (timestamp) ---
Accuracy: XX.XX%
Loss: X.XXXX
Processing Time: XX.XX seconds
Evaluation Time: XX.XX seconds
Additional Metrics:
  metric_name: value
------------------------------------------------
```

2. **JSON File (.json)**
```json
{
    "accuracy": XX.XX,
    "loss": X.XXXX,
    "processing_time": XX.XX,
    "evaluation_time": XX.XX,
    "timestamp": "YYYY-MM-DD HH:MM:SS",
    "additional_metrics": {
        "metric_name": value
    }
}
```

## Integration with FL Core

The output module is designed to work seamlessly with the Federated Learning core:

```python
from fl_output import FLOutput
from fl_core import FederatedLearning

# Initialize FL
fl = FederatedLearning()
fl.set_num_rounds(5)
fl.set_num_clients(3)
fl.run()

# Evaluate and log results
output = FLOutput()
output.evaluate_model(fl.global_model)
output.log_result("results.log")
```

## Error Handling

The module includes comprehensive error handling for:
- Missing or invalid models
- File I/O operations
- Metric computation
- Invalid input parameters

## Best Practices

1. **File Management**
   - Use timestamp-based filenames to prevent overwriting
   - Keep results organized in the `results_from_output` directory
   - Regularly archive old results

2. **Performance Monitoring**
   - Track both processing and evaluation times
   - Monitor model accuracy across training rounds
   - Use custom metrics for specific use cases

3. **Model Evaluation**
   - Evaluate models after each training round
   - Compare results across different configurations
   - Save models at key checkpoints

## Troubleshooting

Common issues and solutions:

1. **Module Import Error**
   - Ensure the project root is in the Python path
   - Check for proper `__init__.py` files in directories

2. **File Permission Issues**
   - Verify write permissions in the output directory
   - Check disk space availability

3. **Model Compatibility**
   - Ensure models are PyTorch nn.Module instances
   - Verify input/output dimensions match the test dataset

## Contributing

To contribute to the output module:
1. Follow the existing code style
2. Add comprehensive documentation
3. Include unit tests for new features
4. Update this documentation
