"""
Multi-Stock Interactive Dashboard

A Streamlit app for training and visualizing predictions for multiple stocks.

Features:
- Train models for any stock ticker
- Compare multiple stocks
- Dark theme (Apple Stocks inspired)
- Interactive charts and metrics

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.price_loader import PriceDataLoader, normalize_prices
from src.data.splits import split_by_time
from src.models.price_model import LSTMPriceModel
from src.evaluation.metrics import evaluate_model
from src.visualization.visualize_predictions import compute_metrics
from torch.utils.data import TensorDataset, DataLoader
from src.utils.seed import set_seed


# Page configuration
st.set_page_config(
    page_title="Multi-Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #000000;
    }
    .stMetric {
        background-color: #1C1C1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #48484A;
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    .stSelectbox label {
        color: #FFFFFF !important;
    }
    .stButton button {
        background-color: #0A84FF;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #0066CC;
    }
</style>
""", unsafe_allow_html=True)


# Popular stock tickers
POPULAR_STOCKS = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMZN'],
    'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
    'Consumer': ['WMT', 'HD', 'MCD', 'NKE', 'SBUX'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB']
}


def get_model_path(ticker: str) -> str:
    """Get path for model checkpoint."""
    return f"results/multi_stock/{ticker}/checkpoints/best_model.pt"


def get_results_dir(ticker: str) -> str:
    """Get results directory for ticker."""
    return f"results/multi_stock/{ticker}"


def model_exists(ticker: str) -> bool:
    """Check if trained model exists for ticker."""
    return os.path.exists(get_model_path(ticker))


@st.cache_data
def load_stock_data(ticker: str, start_date: str, end_date: str, window_size: int):
    """Load and prepare stock data."""
    
    try:
        loader = PriceDataLoader(ticker, start_date, end_date, window_size)
        
        # Download data
        prices_df = loader.download_data()
        
        # Compute returns
        df_with_returns = loader.compute_returns()
        
        # Create sliding windows
        X, y, dates_windowed = loader.create_sliding_windows(df_with_returns)
        
        # Split data
        splits = split_by_time(X, y, dates_windowed)
        
        # Extract raw prices and dates for visualization
        prices = prices_df['close'].values
        dates = prices_df.index.values
        
        # Return splits, raw prices, and dates for visualization
        return splits, prices, dates, True
        
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None, None, None, False


def train_model(ticker: str, splits: dict, config: dict, progress_placeholder):
    """Train model for a specific ticker."""
    
    set_seed(42)
    
    X_train, y_train, dates_train = splits['train']
    X_val, y_val, dates_val = splits['val']
    X_test, y_test, dates_test = splits['test']
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, _ = normalize_prices(
        X_train, X_val, X_test
    )
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_norm),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_norm),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMPriceModel(
        input_dim=1,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    model.to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update progress
        progress = (epoch + 1) / config['num_epochs']
        progress_placeholder.progress(
            progress,
            text=f"Epoch {epoch+1}/{config['num_epochs']} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    results_dir = get_results_dir(ticker)
    os.makedirs(f"{results_dir}/checkpoints", exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'ticker': ticker
    }
    
    torch.save(checkpoint, get_model_path(ticker))
    
    return model, device, train_losses, val_losses


@st.cache_resource
def load_trained_model(ticker: str, config: dict):
    """Load a pre-trained model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMPriceModel(
        input_dim=1,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    checkpoint = torch.load(get_model_path(ticker), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device, checkpoint.get('train_losses', []), checkpoint.get('val_losses', [])


def generate_predictions(model, device, splits, config, all_prices, all_dates):
    """Generate predictions on test set and convert to prices."""
    
    X_train, y_train, _ = splits['train']
    X_val, y_val, _ = splits['val']
    X_test, y_test, dates_test = splits['test']
    
    # Normalize
    X_train_norm, X_val_norm, X_test_norm, _ = normalize_prices(
        X_train, X_val, X_test
    )
    
    # Create test dataloader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_norm),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Generate predictions
    predicted_returns, actual_returns = evaluate_model(model, test_loader, device)
    
    # Convert returns back to prices
    # Find the actual prices corresponding to test dates
    test_prices = []
    predicted_prices = []
    
    for i, test_date in enumerate(dates_test):
        # Find index in all_dates
        date_idx = np.where(all_dates == test_date)[0]
        if len(date_idx) > 0:
            idx = date_idx[0]
            if idx > 0:
                # Current actual price
                current_price = all_prices[idx]
                test_prices.append(current_price)
                
                # Calculate predicted price from previous price and predicted return
                prev_price = all_prices[idx - 1]
                predicted_price = prev_price * (1 + predicted_returns[i])
                predicted_prices.append(predicted_price)
    
    test_prices = np.array(test_prices)
    predicted_prices = np.array(predicted_prices)
    
    return predicted_returns, actual_returns, dates_test, test_prices, predicted_prices


def plot_stock_prices(df, ticker):
    """Create interactive stock price chart like real stock apps."""
    
    fig = go.Figure()
    
    # Calculate price change and direction
    start_price = df['Actual_Price'].iloc[0]
    end_price = df['Actual_Price'].iloc[-1]
    price_change = end_price - start_price
    pct_change = (price_change / start_price) * 100
    direction = "up" if price_change >= 0 else "down"
    color_main = '#30D158' if direction == "up" else '#FF453A'
    
    # Actual price line with area fill
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual_Price'],
        name='Actual',
        line=dict(color=color_main, width=2.5),
        fill='tozeroy',
        fillcolor=f'rgba({int(color_main[1:3], 16)}, {int(color_main[3:5], 16)}, {int(color_main[5:7], 16)}, 0.1)',
        mode='lines',
        hovertemplate='<b>Actual Price</b><br>Date: %{x}<br>$%{y:.2f}<extra></extra>'
    ))
    
    # Predicted price line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted_Price'],
        name='Model Predicted',
        line=dict(color='#0A84FF', width=2, dash='dot'),
        mode='lines',
        hovertemplate='<b>Model Predicted</b><br>Date: %{x}<br>$%{y:.2f}<extra></extra>'
    ))
    
    # Naive baseline (yesterday's price)
    naive_prices = np.concatenate([[df['Actual_Price'].iloc[0]], df['Actual_Price'].values[:-1]])
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=naive_prices,
        name='Naive (Yesterday)',
        line=dict(color='#FF9F0A', width=1.5, dash='dash'),
        mode='lines',
        opacity=0.7,
        hovertemplate='<b>Naive Baseline</b><br>Date: %{x}<br>$%{y:.2f}<extra></extra>'
    ))
    
    # Layout with dark theme - stock app style
    fig.update_layout(
        title=dict(
            text=f'<b style="font-size:20px;">{ticker}</b><br><span style="font-size:26px;">${end_price:.2f}</span> <span style="color:{color_main};font-size:16px;">{price_change:+.2f} ({pct_change:+.2f}%)</span>',
            font=dict(size=18, color='#FFFFFF', family='SF Pro Display, Arial'),
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top'
        ),
        xaxis=dict(
            title='',
            color='#8E8E93',
            gridcolor='#1C1C1E',
            showgrid=False,
            zeroline=False,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(label="All", step="all")
                ],
                bgcolor='#1C1C1E',
                activecolor='#0A84FF',
                font=dict(color='#FFFFFF', size=11),
                x=0.0,
                xanchor='left',
                y=1.10
            )
        ),
        yaxis=dict(
            title='',
            color='#8E8E93',
            gridcolor='#1C1C1E',
            showgrid=True,
            gridwidth=0.5,
            zeroline=False,
            tickprefix='$',
            side='right'
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        hovermode='x unified',
        legend=dict(
            bgcolor='#1C1C1E',
            bordercolor='#48484A',
            borderwidth=1,
            font=dict(color='#FFFFFF', size=12),
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=650,
        margin=dict(t=120, b=50, l=50, r=80)
    )
    
    return fig


def plot_returns_comparison(df, ticker):
    """Create return comparison chart."""
    
    fig = go.Figure()
    
    # Actual returns line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Actual_Return'],
        name='Actual',
        line=dict(color='#30D158', width=2.5),
        mode='lines',
        hovertemplate='<b>Actual</b><br>Date: %{x}<br>Return: %{y:.4f}<extra></extra>'
    ))
    
    # Predicted returns line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Predicted_Return'],
        name='Predicted',
        line=dict(color='#0A84FF', width=2, dash='dash'),
        mode='lines',
        hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Return: %{y:.4f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(
        y=0,
        line=dict(color='#48484A', width=1, dash='dot'),
        opacity=0.5
    )
    
    # Layout with dark theme
    fig.update_layout(
        title=dict(
            text=f'{ticker} Daily Returns',
            font=dict(size=14, color='#FFFFFF', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            color='#8E8E93',
            gridcolor='#1C1C1E',
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title='Return',
            color='#8E8E93',
            gridcolor='#1C1C1E',
            showgrid=True,
            zeroline=True,
            zerolinecolor='#48484A',
            tickformat='.2%'
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        hovermode='x unified',
        legend=dict(
            bgcolor='#1C1C1E',
            bordercolor='#48484A',
            borderwidth=1,
            font=dict(color='#FFFFFF')
        ),
        height=400
    )
    
    return fig


def plot_scatter(actuals, predictions):
    """Create interactive scatter plot."""
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=actuals,
        y=predictions,
        mode='markers',
        marker=dict(
            color='#0A84FF',
            size=6,
            opacity=0.6,
            line=dict(color='#30D158', width=0.5)
        ),
        name='Predictions',
        hovertemplate='<b>Prediction</b><br>Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>'
    ))
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#FF453A', width=2, dash='dash'),
        name='Perfect',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text='Predicted vs Actual',
            font=dict(size=14, color='#FFFFFF', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Actual',
            color='#FFFFFF',
            gridcolor='#1C1C1E',
            showgrid=True
        ),
        yaxis=dict(
            title='Predicted',
            color='#FFFFFF',
            gridcolor='#1C1C1E',
            showgrid=True
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        legend=dict(
            bgcolor='#1C1C1E',
            bordercolor='#48484A',
            borderwidth=1,
            font=dict(color='#FFFFFF')
        ),
        height=500
    )
    
    return fig


def plot_error_distribution(errors):
    """Create interactive error distribution histogram."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=30,
        marker=dict(
            color='#FF453A',
            opacity=0.7,
            line=dict(color='#000000', width=1)
        ),
        name='Errors',
        hovertemplate='Error Range: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Zero line
    fig.add_vline(
        x=0,
        line=dict(color='#30D158', width=2, dash='dash')
    )
    
    fig.update_layout(
        title=dict(
            text='Prediction Errors',
            font=dict(size=14, color='#FFFFFF', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Error',
            color='#FFFFFF',
            gridcolor='#1C1C1E',
            showgrid=True
        ),
        yaxis=dict(
            title='Frequency',
            color='#FFFFFF',
            gridcolor='#1C1C1E',
            showgrid=True
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        showlegend=False,
        height=500
    )
    
    return fig


def plot_training_history(train_losses, val_losses):
    """Create interactive training history plot."""
    
    fig = go.Figure()
    
    epochs = list(range(1, len(train_losses) + 1))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses,
        name='Train Loss',
        line=dict(color='#30D158', width=2),
        mode='lines',
        hovertemplate='<b>Train</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses,
        name='Val Loss',
        line=dict(color='#0A84FF', width=2),
        mode='lines',
        hovertemplate='<b>Validation</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Training & Validation Loss',
            font=dict(size=14, color='#FFFFFF', family='Arial'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Epoch',
            color='#FFFFFF',
            gridcolor='#1C1C1E',
            showgrid=True
        ),
        yaxis=dict(
            title='Loss (MSE)',
            color='#FFFFFF',
            gridcolor='#1C1C1E',
            showgrid=True
        ),
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF'),
        hovermode='x unified',
        legend=dict(
            bgcolor='#1C1C1E',
            bordercolor='#48484A',
            borderwidth=1,
            font=dict(color='#FFFFFF')
        ),
        height=400
    )
    
    return fig



def main():
    """Main Streamlit app."""
    
    # Header
    st.title("📈 Multi-Stock Prediction Dashboard")
    st.markdown("Train and visualize stock return predictions for any ticker")
    st.markdown("---")
    
    # Sidebar - Stock Selection
    st.sidebar.header("🎯 Stock Selection")
    
    # Category selection
    category = st.sidebar.selectbox(
        "Select Category",
        options=list(POPULAR_STOCKS.keys())
    )
    
    # Ticker selection
    selected_ticker = st.sidebar.selectbox(
        "Select Stock",
        options=POPULAR_STOCKS[category],
        help="Choose from popular stocks or enter custom ticker below"
    )
    
    # Custom ticker option
    use_custom = st.sidebar.checkbox("Use custom ticker")
    if use_custom:
        selected_ticker = st.sidebar.text_input(
            "Enter Ticker Symbol",
            value=selected_ticker,
            help="Enter any valid stock ticker (e.g., AAPL, TSLA, etc.)"
        ).upper()
    
    st.sidebar.markdown("---")
    
    # Configuration
    st.sidebar.header("⚙️ Configuration")
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=pd.to_datetime("2020-01-01"),
        help="Historical data start date"
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=pd.to_datetime("2026-02-01"),
        help="Historical data end date (up to yesterday)"
    )
    
    window_size = st.sidebar.slider(
        "Window Size",
        min_value=5,
        max_value=60,
        value=20,
        help="Number of past days to use for prediction"
    )
    
    with st.sidebar.expander("🔧 Advanced Settings"):
        num_epochs = st.slider("Training Epochs", 10, 100, 50)
        batch_size = st.slider("Batch Size", 16, 128, 32)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )
        hidden_dim = st.slider("LSTM Hidden Dimension", 32, 128, 64)
        num_layers = st.slider("LSTM Layers", 1, 4, 2)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2)
    
    config = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'window_size': window_size
    }
    
    st.sidebar.markdown("---")
    
    # Check if model exists
    model_trained = model_exists(selected_ticker)
    
    if model_trained:
        st.sidebar.success(f"✅ Trained model found for {selected_ticker}")
        retrain = st.sidebar.button("🔄 Retrain Model", use_container_width=True)
    else:
        st.sidebar.info(f"ℹ️ No trained model for {selected_ticker}")
        retrain = st.sidebar.button("🚀 Train New Model", use_container_width=True, type="primary")
    
    # Main content area
    st.header(f"📊 {selected_ticker} Stock Analysis")
    
    # Load data
    with st.spinner(f"Loading data for {selected_ticker}..."):
        splits, all_prices, all_dates, success = load_stock_data(
            selected_ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            window_size
        )
    
    if not success or splits is None:
        st.error("Failed to load data. Please check the ticker symbol and date range.")
        return
    
    # Display data info
    _, y_train, dates_train = splits['train']
    _, y_val, dates_val = splits['val']
    _, y_test, dates_test = splits['test']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Samples", len(y_train))
    with col2:
        st.metric("Val Samples", len(y_val))
    with col3:
        st.metric("Test Samples", len(y_test))
    with col4:
        st.metric("Window Size", window_size)
    
    st.markdown("---")
    
    # Training/Loading section
    if retrain or not model_trained:
        st.subheader(f"🔄 Training Model for {selected_ticker}")
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            st.info("Training in progress... This may take a few minutes.")
        
        try:
            model, device, train_losses, val_losses = train_model(
                selected_ticker,
                splits,
                config,
                progress_placeholder
            )
            
            status_placeholder.success(f"✅ Model trained successfully for {selected_ticker}!")
            model_trained = True
            
        except Exception as e:
            status_placeholder.error(f"❌ Training failed: {str(e)}")
            return
    
    # Load model and generate predictions
    if model_trained:
        st.subheader(f"📈 Predictions for {selected_ticker}")
        
        with st.spinner("Loading model and generating predictions..."):
            try:
                model, device, train_losses, val_losses = load_trained_model(selected_ticker, config)
                predicted_returns, actual_returns, dates_test, actual_prices, predicted_prices = generate_predictions(
                    model, device, splits, config, all_prices, all_dates
                )
                metrics = compute_metrics(actual_returns, predicted_returns)
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return
        
        # Display metrics
        st.markdown("### 📊 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MSE", f"{metrics['mse']:.6f}")
        
        with col2:
            st.metric("MAE", f"{metrics['mae']*100:.2f}%")
        
        with col3:
            st.metric("RMSE", f"{metrics['rmse']*100:.2f}%")
        
        with col4:
            dir_acc = metrics['directional_accuracy']
            delta_color = "normal" if dir_acc > 0.5 else "inverse"
            st.metric(
                "Directional Accuracy",
                f"{dir_acc*100:.2f}%",
                delta=f"{(dir_acc-0.5)*100:+.2f}% vs random",
                delta_color=delta_color
            )
        
        # Interpretation with context
        if dir_acc > 0.55:
            st.success("✅ Model performs significantly better than random!")
        elif dir_acc > 0.50:
            st.warning("⚠️ **Model is barely better than random (50% = coin flip)**\n\n"
                      "The graph may LOOK accurate because stock prices are autocorrelated "
                      "(today ≈ yesterday). But the model can't actually predict direction/trends. "
                      "See the **orange 'Naive'** line - just predicting yesterday's price looks similar!")
        else:
            st.error("❌ Model performs worse than random. Price-only data is insufficient.")
        
        st.markdown("---")
        
        # Main visualization - Stock Price Chart
        st.markdown("### 📈 Stock Price Chart")
        
        df = pd.DataFrame({
            'Date': dates_test,
            'Actual_Price': actual_prices,
            'Predicted_Price': predicted_prices,
            'Actual_Return': actual_returns,
            'Predicted_Return': predicted_returns
        })
        
        fig = plot_stock_prices(df, selected_ticker)
        st.plotly_chart(fig, width='stretch')
        
        # Show movement direction indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price_change = actual_prices[-1] - actual_prices[0]
            pct_change = (price_change / actual_prices[0]) * 100
            st.metric(
                "Period Change",
                f"${price_change:.2f}",
                f"{pct_change:.2f}%",
                delta_color="normal"
            )
        
        with col2:
            avg_price = actual_prices.mean()
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col3:
            volatility = actual_returns.std() * 100
            st.metric("Volatility (σ)", f"{volatility:.2f}%")
        
        with col4:
            # Calculate prediction lag/difference
            price_diff = np.abs(predicted_prices - actual_prices).mean()
            price_diff_pct = (price_diff / actual_prices.mean()) * 100
            
            # Also calculate naive baseline error
            naive_prices = np.concatenate([[actual_prices[0]], actual_prices[:-1]])
            naive_diff = np.abs(naive_prices - actual_prices).mean()
            
            st.metric("Avg Price Error", f"${price_diff:.2f}", f"{price_diff_pct:.2f}%")
            st.caption(f"Naive baseline: ${naive_diff:.2f}")
        
        # Returns comparison (collapsible)
        with st.expander("📊 View Returns Analysis"):
            fig_returns = plot_returns_comparison(df, selected_ticker)
            st.plotly_chart(fig_returns, width='stretch')
        
        # Debug: Show prediction logic verification
        with st.expander("🔍 Prediction Verification (Debug Info)"):
            st.markdown("""
            **Verifying No Data Leakage:**
            - Model inputs: Day 1 to Day 20 (prices)
            - Model predicts: Day 21 (return)
            - Predicted price for Day 21 = Day 20 price × (1 + predicted return)
            """)
            
            # Show sample of how predictions work
            sample_df = df.head(10).copy()
            sample_df['Price_Error_$'] = sample_df['Predicted_Price'] - sample_df['Actual_Price']
            sample_df['Price_Error_%'] = (sample_df['Price_Error_$'] / sample_df['Actual_Price']) * 100
            
            st.dataframe(
                sample_df[['Date', 'Actual_Price', 'Predicted_Price', 'Price_Error_$', 'Price_Error_%']].style.format({
                    'Actual_Price': '${:.2f}',
                    'Predicted_Price': '${:.2f}',
                    'Price_Error_$': '${:.2f}',
                    'Price_Error_%': '{:.2f}%'
                }),
                use_container_width=True
            )
            
            st.markdown(f"""
            **Key Metrics:**
            - Mean Absolute Error: ${np.abs(predicted_prices - actual_prices).mean():.2f}
            - RMSE: ${np.sqrt(((predicted_prices - actual_prices)**2).mean()):.2f}
            - Correlation: {np.corrcoef(actual_prices, predicted_prices)[0,1]:.4f}
            
            ⚠️ **If correlation is very high (>0.95), the model might just be predicting "yesterday's price"**
            """)
        
        st.markdown("---")
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Price Prediction Accuracy")
            fig2 = plot_scatter(actual_prices, predicted_prices)
            st.plotly_chart(fig2, width='stretch')
        
        with col2:
            st.markdown("### 📊 Prediction Error Distribution")
            price_errors = predicted_prices - actual_prices
            fig3 = plot_error_distribution(price_errors)
            st.plotly_chart(fig3, width='stretch')
        
        # Training history
        if train_losses and val_losses:
            st.markdown("---")
            st.markdown("### 📈 Training History")
            fig4 = plot_training_history(train_losses, val_losses)
            st.plotly_chart(fig4, width='stretch')
        
        # Data table
        st.markdown("---")
        with st.expander("📋 View Raw Predictions"):
            display_df = df.copy()
            display_df['Price_Error'] = display_df['Predicted_Price'] - display_df['Actual_Price']
            display_df['Price_Error_Pct'] = (display_df['Price_Error'] / display_df['Actual_Price']) * 100
            display_df['Correct_Direction'] = (
                (display_df['Predicted_Return'] > 0) == (display_df['Actual_Return'] > 0)
            )
            
            st.dataframe(
                display_df.style.format({
                    'Actual_Price': '${:.2f}',
                    'Predicted_Price': '${:.2f}',
                    'Price_Error': '${:.2f}',
                    'Price_Error_Pct': '{:.2f}%',
                    'Actual_Return': '{:.4f}',
                    'Predicted_Return': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"{selected_ticker}_predictions.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #48484A;'>"
        "Multi-Stock Prediction Dashboard | Built with Streamlit & PyTorch"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
