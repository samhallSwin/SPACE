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
        accuracy = metrics.get('accuracy', 0)
        # Correctly extract round_times from additional_metrics
        round_times = metrics.get('additional_metrics', {}).get('round_times', {})
        # Generate plots and dashboard under the same run folder
        self._create_training_progress_plot(metrics_file, round_times, accuracy)
        self._create_dashboard(metrics_file, metrics)
        # If additional run JSONs exist, optional comparison can be invoked separately

    def _create_training_progress_plot(self, metrics_file: str, round_times: dict, accuracy: float):
        """Create interactive training progress visualization"""
        if not round_times:
            return
        runs_folder = os.path.dirname(metrics_file)
        rounds = list(range(1, len(round_times) + 1))
        times = list(round_times.values())
        # Build Plotly figure
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Round Processing Times','Model Performance'), vertical_spacing=0.15)
        fig.add_trace(go.Bar(x=rounds, y=times, name='Processing Time', marker_color='lightblue'), row=1, col=1)
        fig.add_trace(go.Scatter(x=['Final'], y=[accuracy], mode='markers', marker=dict(size=20, color='green'), name=f'Accuracy: {accuracy:.2f}%'), row=2, col=1)
        fig.update_layout(title='Federated Learning Training Progress', showlegend=True, height=600)
        out_path = os.path.join(runs_folder, 'training_progress.html')
        fig.write_html(out_path)
        print(f"Training progress saved to {out_path}")

    def _create_dashboard(self, metrics_file: str, metrics: dict):
        """Create a comprehensive HTML dashboard of metrics."""
        runs_folder = os.path.dirname(metrics_file)
        timestamp = metrics.get('timestamp', '')
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
            <div class="metric-value">{metrics.get('accuracy',0):.2f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Final Loss</div>
            <div class="metric-value">{metrics.get('loss',0):.4f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Training Time</div>
            <div class="metric-value">{metrics.get('processing_time',0):.2f}s</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Number of Rounds</div>
            <div class="metric-value">{len(metrics.get('round_times',{}))}</div>
        </div>
    </div>
    <h2>Round Processing Times</h2>
    <ul>
"""
        for r, t in metrics.get('round_times',{}).items():
            html += f"        <li>{r}: {t:.2f} seconds</li>\n"
        html += "    </ul>\n</body>\n</html>"
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
