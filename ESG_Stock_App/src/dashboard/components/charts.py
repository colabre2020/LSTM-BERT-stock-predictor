import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Dict
import numpy as np
from datetime import datetime

def plot_stock_prices(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_moving_average(data, window=20):
    data['Moving Average'] = data['Close'].rolling(window=window).mean()
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], data['Moving Average'], label=f'{window}-Day Moving Average', color='orange')
    plt.title('Stock Prices with Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_volume(data):
    plt.figure(figsize=(12, 6))
    plt.bar(data['Date'], data['Volume'], color='gray')
    plt.title('Stock Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.grid()
    plt.show()

def create_stock_chart(predictions: List[Dict], stock_name: str) -> go.Figure:
    # Handle empty predictions
    if not predictions:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            title_text=f"{stock_name} Stock Price Prediction (No data available)",
            height=400,  # Reduced height
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40),  # Compact margins
        )
        return fig

    dates = [datetime.fromisoformat(p['predictionDate']) for p in predictions]
    predicted_prices = [p['predictedPrice'] for p in predictions]
    actual_prices = [p['actualPrice'] for p in predictions]
    
    fig = make_subplots(rows=1, cols=1)
    
    # Add price traces
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual_prices,
            name='Actual Price',
            line=dict(color='#1f77b4', width=2),
            mode='lines'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted_prices,
            name='Predicted Price',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            mode='lines'
        )
    )
    
    # Add confidence intervals if available
    if predictions and 'confidence' in predictions[0]:
        confidences = [p['confidence'] for p in predictions]
        upper_bound = [p + c for p, c in zip(predicted_prices, confidences)]
        lower_bound = [p - c for p, c in zip(predicted_prices, confidences)]
        
        fig.add_trace(
            go.Scatter(
                x=dates + dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255,127,14,0.1)',
                line=dict(color='rgba(255,127,14,0)'),
                name='Confidence Interval',
                showlegend=True
            )
        )
    
    # Update layout for a more compact and modern look
    fig.update_layout(
        title=dict(
            text=f"{stock_name} Stock Price Prediction",
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        height=400,  # Reduced height
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),  # Compact margins
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
        plot_bgcolor='white'
    )
    
    return fig

def create_prediction_accuracy_chart(predictions: List[Dict]) -> go.Figure:
    if not predictions:
        fig = go.Figure()
        fig.update_layout(
            title_text="Prediction Accuracy Over Time (No data available)",
            height=300,  # More compact
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

    dates = [datetime.fromisoformat(p['predictionDate']) for p in predictions]
    accuracies = []
    
    for pred in predictions:
        accuracy = abs(pred['predictedPrice'] - pred['actualPrice']) / pred['actualPrice'] * 100
        accuracies.append(100 - accuracy)  # Convert error to accuracy percentage
    
    # Calculate moving average
    window = min(5, len(accuracies))
    if window > 0:
        ma = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        ma_dates = dates[window-1:]
    else:
        ma = []
        ma_dates = []
    
    fig = go.Figure()
    
    # Add accuracy scatter plot
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=accuracies,
            name='Accuracy',
            mode='markers',
            marker=dict(
                size=6,
                color=accuracies,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(
                    title='Accuracy %',
                    thickness=10,
                    len=0.8,
                )
            )
        )
    )
    
    # Add moving average line if we have enough data
    if len(ma) > 0:
        fig.add_trace(
            go.Scatter(
                x=ma_dates,
                y=ma,
                name=f'{window}-Day MA',
                line=dict(color='black', width=1, dash='dot')
            )
        )
    
    # Update layout for a more compact and modern look
    fig.update_layout(
        title=dict(
            text="Prediction Accuracy Over Time",
            font=dict(size=14),
            x=0.5,
            xanchor='center'
        ),
        height=300,  # More compact
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
        plot_bgcolor='white'
    )
    
    return fig

def create_model_comparison_chart(model_metrics: Dict[str, Dict]) -> go.Figure:
    models = list(model_metrics.keys())
    metrics = ['MAE', 'RMSE', 'R2']
    
    fig = go.Figure()
    
    for metric in metrics:
        values = [model_metrics[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.2f}' for v in values],
                textposition='auto',
            )
        )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    return fig

def create_correlation_heatmap(predictions: List[Dict]) -> go.Figure:
    """Create a heatmap showing correlations between different stocks"""
    if not predictions:
        fig = go.Figure()
        fig.update_layout(
            title='Stock Price Correlation Matrix (No data available)',
            height=600,
            width=600
        )
        return fig

    # Extract unique stock names
    stock_names = list(set(p['stockName'] for p in predictions))
    n_stocks = len(stock_names)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_stocks, n_stocks))
    
    for i, stock1 in enumerate(stock_names):
        for j, stock2 in enumerate(stock_names):
            stock1_prices = [p['actualPrice'] for p in predictions if p['stockName'] == stock1]
            stock2_prices = [p['actualPrice'] for p in predictions if p['stockName'] == stock2]
            
            if len(stock1_prices) == len(stock2_prices) and len(stock1_prices) > 0:
                correlation = np.corrcoef(stock1_prices, stock2_prices)[0, 1]
                corr_matrix[i, j] = correlation
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=stock_names,
        y=stock_names,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=[[f'{val:.2f}' for val in row] for row in corr_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Stock Price Correlation Matrix',
        height=600,
        width=600
    )
    
    return fig