"""
Stock Prediction Visualization

Purpose:
    - Visualize actual vs predicted returns
    - Display model performance metrics
    - Apple Stocks app inspired dark theme
    - Clean, presentation-ready plots

This is for interpretation and debugging only.
NO training, NO model changes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, Tuple, Optional
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def load_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Create a DataFrame from prediction results.
    
    Args:
        predictions: Model predictions (returns)
        actuals: Actual returns
        dates: Corresponding dates
        
    Returns:
        DataFrame with columns: date, actual, predicted
    """
    df = pd.DataFrame({
        'date': dates,
        'actual': actuals,
        'predicted': predictions
    })
    return df


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        actuals: True values
        predictions: Predicted values
        
    Returns:
        Dictionary with MSE, MAE, RMSE, Directional Accuracy
    """
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    # Directional accuracy
    correct_direction = np.sign(predictions) == np.sign(actuals)
    dir_acc = np.mean(correct_direction)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': dir_acc
    }


def create_stock_visualization(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    ticker: str = "STOCK",
    save_path: Optional[str] = None
):
    """
    Create Apple Stocks app inspired visualization.
    
    Features:
    - Dark theme
    - Actual vs Predicted comparison
    - Metrics display
    - Clean, professional styling
    
    Args:
        df: DataFrame with columns [date, actual, predicted]
        metrics: Dictionary with performance metrics
        ticker: Stock ticker symbol
        save_path: Optional path to save figure
    """
    # Set dark theme style
    plt.style.use('dark_background')
    
    # Create figure with custom dark background
    fig = plt.figure(figsize=(16, 10), facecolor='#000000')
    
    # Create main plot area
    ax = plt.subplot(111, facecolor='#000000')
    
    # Colors inspired by Apple Stocks app
    color_actual = '#30D158'      # Green for actual (Apple green)
    color_predicted = '#0A84FF'   # Blue for predicted (Apple blue)
    color_grid = '#1C1C1E'        # Subtle grid
    color_text = '#FFFFFF'        # White text
    
    # Plot actual values
    ax.plot(
        df['date'], 
        df['actual'], 
        color=color_actual,
        linewidth=2.5,
        label='Actual Return',
        alpha=0.9,
        zorder=3
    )
    
    # Plot predicted values
    ax.plot(
        df['date'], 
        df['predicted'], 
        color=color_predicted,
        linewidth=2,
        linestyle='--',
        label='Predicted Return',
        alpha=0.85,
        zorder=2
    )
    
    # Add zero line
    ax.axhline(y=0, color='#48484A', linewidth=1, linestyle='-', alpha=0.5, zorder=1)
    
    # Styling
    ax.set_xlabel('Date', fontsize=13, color=color_text, fontweight='500')
    ax.set_ylabel('Return', fontsize=13, color=color_text, fontweight='500')
    ax.set_title(
        f'{ticker} Stock Prediction Analysis',
        fontsize=20,
        color=color_text,
        fontweight='600',
        pad=20
    )
    
    # Grid
    ax.grid(True, alpha=0.15, color=color_grid, linewidth=1)
    ax.set_axisbelow(True)
    
    # Legend
    legend = ax.legend(
        loc='upper left',
        fontsize=11,
        frameon=True,
        facecolor='#1C1C1E',
        edgecolor='#48484A',
        framealpha=0.95
    )
    for text in legend.get_texts():
        text.set_color(color_text)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#48484A')
        spine.set_linewidth(1)
    
    # Tick styling
    ax.tick_params(colors=color_text, labelsize=10)
    
    # Rotate date labels
    plt.xticks(rotation=45, ha='right')
    
    # Add metrics panel (top right)
    metrics_text = (
        f"Model Performance Metrics\n"
        f"{'─' * 35}\n"
        f"MSE:                 {metrics['mse']:.6f}\n"
        f"MAE:                 {metrics['mae']:.6f} ({metrics['mae']*100:.2f}%)\n"
        f"RMSE:                {metrics['rmse']:.6f} ({metrics['rmse']*100:.2f}%)\n"
        f"Directional Acc:     {metrics['directional_accuracy']:.4f} ({metrics['directional_accuracy']*100:.2f}%)\n"
    )
    
    # Add text box with metrics
    props = dict(
        boxstyle='round,pad=0.8',
        facecolor='#1C1C1E',
        edgecolor='#48484A',
        alpha=0.95,
        linewidth=1.5
    )
    
    ax.text(
        0.98, 0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=props,
        color=color_text,
        family='monospace',
        fontweight='500'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            facecolor='#000000',
            edgecolor='none',
            bbox_inches='tight'
        )
        print(f"Visualization saved to: {save_path}")
    
    return fig


def create_detailed_analysis(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    ticker: str = "STOCK",
    save_path: Optional[str] = None
):
    """
    Create comprehensive multi-panel analysis.
    
    Panels:
    1. Actual vs Predicted time series
    2. Scatter plot (correlation)
    3. Error distribution
    4. Rolling directional accuracy
    
    Args:
        df: DataFrame with predictions
        metrics: Performance metrics
        ticker: Stock ticker
        save_path: Save location
    """
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(18, 12), facecolor='#000000')
    
    # Colors
    color_actual = '#30D158'
    color_predicted = '#0A84FF'
    color_error = '#FF453A'
    color_text = '#FFFFFF'
    
    # Panel 1: Time series
    ax1 = plt.subplot(2, 2, 1, facecolor='#000000')
    ax1.plot(df['date'], df['actual'], color=color_actual, linewidth=2, label='Actual', alpha=0.9)
    ax1.plot(df['date'], df['predicted'], color=color_predicted, linewidth=2, linestyle='--', label='Predicted', alpha=0.85)
    ax1.axhline(y=0, color='#48484A', linewidth=1, alpha=0.5)
    ax1.set_title(f'{ticker} - Returns Over Time', fontsize=14, color=color_text, fontweight='600')
    ax1.set_xlabel('Date', fontsize=11, color=color_text)
    ax1.set_ylabel('Return', fontsize=11, color=color_text)
    ax1.grid(True, alpha=0.15, color='#1C1C1E')
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1C1C1E', edgecolor='#48484A')
    ax1.tick_params(colors=color_text)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel 2: Scatter plot
    ax2 = plt.subplot(2, 2, 2, facecolor='#000000')
    ax2.scatter(df['actual'], df['predicted'], alpha=0.5, s=30, color=color_predicted, edgecolors=color_actual, linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(df['actual'].min(), df['predicted'].min())
    max_val = max(df['actual'].max(), df['predicted'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], color=color_error, linestyle='--', linewidth=2, label='Perfect Prediction', alpha=0.7)
    
    ax2.set_title('Predicted vs Actual', fontsize=14, color=color_text, fontweight='600')
    ax2.set_xlabel('Actual Return', fontsize=11, color=color_text)
    ax2.set_ylabel('Predicted Return', fontsize=11, color=color_text)
    ax2.grid(True, alpha=0.15, color='#1C1C1E')
    ax2.legend(loc='upper left', fontsize=10, facecolor='#1C1C1E', edgecolor='#48484A')
    ax2.tick_params(colors=color_text)
    
    # Panel 3: Error distribution
    ax3 = plt.subplot(2, 2, 3, facecolor='#000000')
    errors = df['predicted'] - df['actual']
    ax3.hist(errors, bins=50, color=color_error, alpha=0.7, edgecolor='#000000', linewidth=0.5)
    ax3.axvline(x=0, color=color_actual, linewidth=2, linestyle='--', alpha=0.8)
    ax3.set_title('Prediction Error Distribution', fontsize=14, color=color_text, fontweight='600')
    ax3.set_xlabel('Error (Predicted - Actual)', fontsize=11, color=color_text)
    ax3.set_ylabel('Frequency', fontsize=11, color=color_text)
    ax3.grid(True, alpha=0.15, color='#1C1C1E')
    ax3.tick_params(colors=color_text)
    
    # Panel 4: Rolling directional accuracy
    ax4 = plt.subplot(2, 2, 4, facecolor='#000000')
    window = 50
    correct = (np.sign(df['predicted']) == np.sign(df['actual'])).astype(int)
    rolling_acc = pd.Series(correct).rolling(window=window, min_periods=1).mean()
    
    ax4.plot(df['date'], rolling_acc, color=color_predicted, linewidth=2)
    ax4.axhline(y=0.5, color=color_error, linewidth=2, linestyle='--', label='Random (50%)', alpha=0.7)
    ax4.set_title(f'Rolling Directional Accuracy (Window={window})', fontsize=14, color=color_text, fontweight='600')
    ax4.set_xlabel('Date', fontsize=11, color=color_text)
    ax4.set_ylabel('Accuracy', fontsize=11, color=color_text)
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.15, color='#1C1C1E')
    ax4.legend(loc='upper left', fontsize=10, facecolor='#1C1C1E', edgecolor='#48484A')
    ax4.tick_params(colors=color_text)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Overall title
    fig.suptitle(
        f'{ticker} Stock Prediction - Detailed Analysis',
        fontsize=18,
        color=color_text,
        fontweight='600',
        y=0.98
    )
    
    # Metrics summary at bottom
    metrics_text = (
        f"MSE: {metrics['mse']:.6f}  |  "
        f"MAE: {metrics['mae']:.6f} ({metrics['mae']*100:.2f}%)  |  "
        f"RMSE: {metrics['rmse']:.6f} ({metrics['rmse']*100:.2f}%)  |  "
        f"Directional Accuracy: {metrics['directional_accuracy']:.4f} ({metrics['directional_accuracy']*100:.2f}%)"
    )
    
    fig.text(
        0.5, 0.02,
        metrics_text,
        ha='center',
        fontsize=11,
        color=color_text,
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#1C1C1E', edgecolor='#48484A', alpha=0.95)
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            facecolor='#000000',
            edgecolor='none',
            bbox_inches='tight'
        )
        print(f"Detailed analysis saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Demo: Load results from training and visualize
    print("Loading prediction results...")
    
    # This would typically load from saved results
    # For now, create synthetic example
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    actuals = np.random.randn(100) * 0.02
    predictions = actuals + np.random.randn(100) * 0.01
    
    # Create DataFrame
    df = load_predictions(predictions, actuals, dates)
    
    # Compute metrics
    metrics = compute_metrics(actuals, predictions)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Simple visualization
    create_stock_visualization(
        df, 
        metrics, 
        ticker="DEMO",
        save_path="demo_visualization.png"
    )
    
    # Detailed analysis
    create_detailed_analysis(
        df,
        metrics,
        ticker="DEMO",
        save_path="demo_detailed_analysis.png"
    )
    
    plt.show()
    
    print("\nVisualization complete!")
