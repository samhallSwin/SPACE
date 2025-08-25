"""
Filename: fl_visualization.py
Description: Visualization module for Federated Learning results
Author: Gagandeep Singh
Version: 1.0
Date: 2025-05-15
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FLVisualization:
    def __init__(self, results_dir: str = "./results_from_output"):
        """
        Initialize visualization utility
        Args:
            results_dir: Root directory where each run folder lives
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def visualize_from_json(self, metrics_file: str):
        """Create visualizations from FL output JSON."""
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        # Extract core metrics
        # Correctly extract round_times from additional_metrics
        # Generate plots and dashboard under the same run folder
        self._create_dashboard(metrics_file, metrics)
        # If additional run JSONs exist, optional comparison can be invoked separately

    def _create_dashboard(self, metrics_file: str, metrics: dict):
        """Create a comprehensive HTML dashboard of metrics and training progress."""
        runs_folder = os.path.dirname(metrics_file)
        timestamp = metrics.get('timestamp', '')
        round_accuracies = metrics.get('additional_metrics', {}).get('round_accuracies', [])
        num_rounds = len(round_accuracies)

        # Build Plotly figure for training progress
        rounds = list(range(1, num_rounds + 1))
        times = list(metrics.get('additional_metrics', {}).get('round_times', {}).values())

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Round Processing Times', 'Round Accuracies'),
            vertical_spacing=0.15
        )

        # Add bar plot for processing times
        fig.add_trace(
            go.Bar(x=rounds, y=times, name='Processing Time', marker_color='lightblue'),
            row=1, col=1
        )

        # Add line plot for accuracies
        fig.add_trace(
            go.Scatter(x=rounds, y=round_accuracies, mode='lines+markers', name='Accuracy', line=dict(color='green', width=2)),
            row=2, col=1
        )

        fig.update_layout(
            title='Federated Learning Training Progress',
            showlegend=True,
            height=700
        )

        # Convert Plotly figure to HTML
        training_progress_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Combine dashboard and training progress
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FL Training Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric-box {{ display: inline-block; margin: 10px; padding: 15px; border: 2px solid #ddd; border-radius: 5px; background-color: #f9f9f9; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        h1 {{ color: #333; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>Federated Learning Results Dashboard</h1>
    <p class="timestamp">Generated: {timestamp}</p>
    <div class="metrics-container">
        <div class="metric-box">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value">{metrics.get('accuracy', 0):.2f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Final Loss</div>
            <div class="metric-value">{metrics.get('loss', 0):.4f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Training Time</div>
            <div class="metric-value">{metrics.get('processing_time', 0):.2f}s</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Number of Rounds</div>
            <div class="metric-value">{num_rounds}</div>
        </div>
    </div>
    <h2>Training Progress</h2>
    {training_progress_html}
</body>
</html>
"""
        dash_path = os.path.join(runs_folder, 'dashboard.html')
        with open(dash_path, 'w') as f:
            f.write(html)
        print(f"Dashboard saved to {dash_path}")

    def plot_model_comparison(self, metrics_files: list):
        """Compare multiple FL training runs using bar plots"""
        accuracies, losses, labels = [], [], []
        for mf in metrics_files:
            with open(mf) as f:
                m = json.load(f)
            accuracies.append(m.get('accuracy',0))
            losses.append(m.get('loss',0))
            labels.append(os.path.basename(m).split('.')[0])
        _, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
        ax1.bar(labels, accuracies, color='skyblue', alpha=0.7)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim(0,100)
        ax2.bar(labels, losses, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss Comparison')
        plt.tight_layout()
        out_img = os.path.join(self.results_dir, 'model_comparison.png')
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison saved to {out_img}")

# Integration with existing FL modules
if __name__ == "__main__":
    import glob
    viz = FLVisualization()
    # find latest run folder
    runs = [d for d in glob.glob(os.path.join(viz.results_dir, '*')) if os.path.isdir(d)]
    if not runs:
        print("No runs found under", viz.results_dir)
    else:
        latest = max(runs, key=os.path.getctime)
        jsons = glob.glob(os.path.join(latest, '*.json'))
        if jsons:
            mf = max(jsons, key=os.path.getctime)
            print(f"Visualizing from {mf}")
            viz.visualize_from_json(mf)
            if len(jsons)>1:
                viz.plot_model_comparison(sorted(jsons)[-3:])
