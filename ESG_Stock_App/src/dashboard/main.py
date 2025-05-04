import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yaml
import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer

from ..data.db_handler import WeaviateHandler
from ..models.trainer import ModelTrainer
from ..models.bert_lstm_model import BERTLSTMStockPredictor
from .components.charts import create_stock_chart, create_prediction_accuracy_chart
from ..utils.device import get_device

# Override config for development
import os
os.environ['DASH_DEBUG'] = 'true'

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Flask-Login
login_manager = LoginManager()

# Create a User class
class User(UserMixin):
    def __init__(self, username):
        self.id = username

# Initialize Dash app with authentication
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server
server.secret_key = config['dashboard']['secret_key']  # Set the secret key from config
login_manager.init_app(server)

# Initialize components with device
device = get_device()
db_handler = WeaviateHandler()
model_trainer = ModelTrainer()

# Load available stocks from Excel
def get_available_stocks():
    try:
        df = pd.read_excel(config['data']['stock_data_path'])
        return sorted(df['Stock'].unique().tolist())
    except Exception as e:
        print(f"Error loading stocks: {e}")
        return ['AAPL', 'GOOGL', 'MSFT']  # Fallback list

# Add new function to generate and load sample data
def load_sample_data():
    """Generate sample data for stocks"""
    stocks = load_stock_list()  # Use the expanded stock list
    print(f"Generating sample data for {len(stocks)} stocks")
    
    for stock in stocks:
        base_price = np.random.uniform(100, 500)
        volatility = np.random.uniform(0.01, 0.03)
        prices = [base_price]
        
        # Generate more realistic price movements
        for _ in range(6):  # Last 7 days
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Store predictions with more realistic data
        for i, price in enumerate(prices):
            date = datetime.now() - timedelta(days=i)
            prediction = float(price * (1 + np.random.uniform(-0.02, 0.02)))
            confidence = float(np.random.uniform(0.85, 0.95))
            
            try:
                db_handler.store_prediction(
                    stock_name=stock,
                    prediction=prediction,
                    confidence=confidence,
                    actual_price=float(price),
                    metadata={
                        "generation_date": date.isoformat(),
                        "volatility": volatility,
                        "market_sentiment": np.random.choice(["bullish", "neutral", "bearish"])
                    }
                )
            except Exception as e:
                print(f"Error storing prediction for {stock}: {e}")
                continue

def load_stock_list():
    """Load stocks from Excel file, Weaviate, or return comprehensive default list"""
    default_stocks = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 
        'V', 'WMT', 'PG', 'JNJ', 'UNH', 'MA', 'HD', 'BAC', 'XOM', 'PFE', 
        'CSCO', 'INTC'
    ]
    
    try:
        # Try to load from Excel file first
        excel_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'STOCK_PRICE_TOP_50.xlsx')
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            if 'Stock' in df.columns:
                stocks = sorted(df['Stock'].unique().tolist())
            else:
                stocks = sorted([col for col in df.columns if col != 'Date'])
            if stocks:
                return stocks
    except Exception as e:
        print(f"Error loading stocks from Excel: {e}")
    
    try:
        # Try to get unique stocks from Weaviate
        predictions = db_handler.get_predictions(limit=1000)
        if predictions:
            stocks = sorted(set(p['stockName'] for p in predictions))
            if stocks:
                return stocks
    except Exception as e:
        print(f"Error loading stocks from Weaviate: {e}")
    
    return default_stocks

# Layout components
login_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("ESG Stock Predictor", className="text-center mb-4", 
                               style={"color": "#2c3e50", "font-weight": "bold"}),
                        html.Hr(className="my-4"),
                        dbc.Form([
                            dbc.Row([
                                dbc.Label("Username", html_for="username", 
                                         className="mb-2", style={"font-size": "1.1rem"}),
                                dbc.Input(
                                    id="username",
                                    type="text",
                                    placeholder="Enter username",
                                    className="mb-4 py-2",
                                    style={
                                        "borderRadius": "8px",
                                        "fontSize": "1.1rem",
                                        "border": "2px solid #e0e0e0"
                                    }
                                ),
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Label("Password", html_for="password", 
                                         className="mb-2", style={"font-size": "1.1rem"}),
                                dbc.Input(
                                    id="password",
                                    type="password",
                                    placeholder="Enter password",
                                    className="mb-4 py-2",
                                    style={
                                        "borderRadius": "8px",
                                        "fontSize": "1.1rem",
                                        "border": "2px solid #e0e0e0"
                                    }
                                ),
                            ], className="mb-3"),
                            dbc.Button(
                                "Sign In",
                                id="login-button",
                                color="primary",
                                className="w-100 mb-3 py-2",
                                style={
                                    "borderRadius": "8px",
                                    "fontSize": "1.2rem",
                                    "fontWeight": "500",
                                    "backgroundColor": "#007bff",
                                    "borderColor": "#007bff",
                                    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
                                }
                            ),
                            html.Div(id="login-error", className="text-danger text-center mt-2")
                        ])
                    ])
                ], className="shadow", style={
                    "borderRadius": "15px",
                    "border": "none",
                    "backgroundColor": "#ffffff",
                    "minWidth": "400px",
                    "padding": "20px"
                })
            ], width={"size": 6, "offset": 3}, className="mt-5")
        ], style={"minHeight": "100vh"}, className="d-flex align-items-center justify-content-center")
    ], fluid=True, style={"backgroundColor": "#f8f9fa"})
])

chat_layout = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("Chat with Model"),
            dbc.CardBody([
                html.Div(id="chat-history", style={"height": "300px", "overflowY": "auto"}),
                dbc.Input(id="chat-input", placeholder="Ask about stock predictions...", type="text"),
                dbc.Button("Send", id="send-chat", color="primary", className="mt-2"),
                dbc.FormText("Example: 'What's your prediction for AAPL?' or 'Why do you think GOOGL will go up?'")
            ])
        ], className="mb-3"),
        dbc.Alert(
            "Sample data is used when real market data is not available. It generates realistic-looking stock prices "
            "and predictions to demonstrate the dashboard's features. Click 'Load Sample Data' to try it out.",
            color="info",
            dismissable=True,
            id="sample-data-info"
        )
    ], width=12)
])

# Update main layout to move chat below performance metrics
main_layout = html.Div([
    # Navbar with reduced padding
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.Button("Load Sample Data", id="load-sample-data", color="success", size="sm", className="mr-2")),
            dbc.NavItem(dbc.Button("Logout", id="logout-button", color="light", size="sm"))
        ],
        brand="Stock Price Dashboard",
        color="primary",
        dark=True,
        className="py-1"
    ),
    dbc.Container([
        dbc.Row([
            dbc.Col([html.Div(id="notification-container", className="mt-2")], width=12),
        ]),
        dbc.Row([
            dbc.Col([
                # Stock selector in a row with title
                dbc.Row([
                    dbc.Col(html.H5("Stock Price Predictions", className="mb-0 mt-2"), width="auto"),
                    dbc.Col(
                        dcc.Dropdown(
                            id='stock-selector',
                            options=[{'label': stock, 'value': stock} for stock in load_stock_list()],
                            value=load_stock_list()[0],
                            className="ml-2"
                        ),
                        width=True
                    )
                ], align="center"),
                # Main chart
                dcc.Loading(
                    id="loading-chart",
                    type="default",
                    children=[dcc.Graph(id='stock-price-chart', config={'displayModeBar': False})],
                ),
            ], width=12)
        ]),
        dbc.Row([
            # Performance metrics and table in a more compact layout
            dbc.Col([
                html.H6("Model Performance", className="mb-2"),
                dcc.Loading(
                    id="loading-accuracy",
                    type="default",
                    children=[dcc.Graph(
                        id='accuracy-chart',
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )],
                ),
            ], width=6),
            dbc.Col([
                html.H6("Recent Predictions", className="mb-2"),
                dcc.Loading(
                    id="loading-table",
                    type="default",
                    children=[
                        html.Div(
                            id='predictions-table',
                            style={'maxHeight': '300px', 'overflowY': 'auto'}
                        )
                    ],
                ),
            ], width=6)
        ]),
        # Move chat layout below performance metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H6("AI Assistant", className="mb-0"),
                        html.Small("Ask about stock predictions, trends, and analysis", className="text-muted")
                    ]),
                    dbc.CardBody([
                        html.Div(id="chat-history", style={
                            "height": "300px",
                            "overflowY": "auto",
                            "backgroundColor": "#f8f9fa",
                            "padding": "15px",
                            "borderRadius": "10px",
                            "marginBottom": "15px"
                        }),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Input(
                                        id="chat-input",
                                        placeholder="Ask about stock predictions...",
                                        type="text",
                                        style={
                                            "borderRadius": "4px",
                                            "height": "38px"
                                        },
                                        n_submit=0  # Enable Enter key submission
                                    ),
                                ], width=10),
                                dbc.Col([
                                    dbc.Button(
                                        "Send",
                                        id="send-chat",
                                        color="primary",
                                        style={
                                            "borderRadius": "4px",
                                            "width": "100%"
                                        },
                                        n_clicks=0
                                    ),
                                ], width=2),
                            ], className="g-0"),
                        ]),
                        dbc.FormText([
                            "Press Enter or click Send. Try asking: ",
                            html.Span("'What's the prediction for AAPL?'", className="fw-bold"),
                            " or ",
                            html.Span("'Why do you expect GOOGL to trend this way?'", className="fw-bold")
                        ], className="mt-2")
                    ])
                ], className="mb-3 shadow-sm")
            ], width=12)
        ]),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,
            n_intervals=0
        ),
        # Store for chat history
        dcc.Store(id='chat-history-store', data=[])
    ], fluid=True, className="py-2")
])

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id='authentication-status')
])

@login_manager.user_loader
def load_user(username):
    return User(username)

@app.callback(
    [Output('authentication-status', 'data'),
     Output('login-error', 'children')],
    [Input('login-button', 'n_clicks')],
    [State('username', 'value'),
     State('password', 'value')]
)
def login(n_clicks, username, password):
    if n_clicks is None:
        return None, ''
    
    # In production, replace with proper authentication
    if username == 'admin' and password == 'admin':
        user = User(username)
        login_user(user)
        return {'authenticated': True}, ''
    return {'authenticated': False}, 'Invalid credentials'

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('authentication-status', 'data')]
)
def display_page(pathname, auth_status):
    if auth_status is None or not auth_status.get('authenticated'):
        return login_layout
    return main_layout

@app.callback(
    Output('authentication-status', 'clear_data'),
    [Input('logout-button', 'n_clicks')]
)
def logout(n_clicks):
    if n_clicks is not None:
        logout_user()
        return True
    return False

# Add callback for sample data loading
@app.callback(
    [Output('load-sample-data', 'disabled'),
     Output('notification-container', 'children')],
    [Input('load-sample-data', 'n_clicks')]
)
def handle_sample_data_load(n_clicks):
    if n_clicks is not None:
        try:
            load_sample_data()
            return True, dbc.Alert(
                "Sample data loaded successfully! You should see the charts updating...",
                color="success",
                duration=5000,
                is_open=True,
            )
        except Exception as e:
            return False, dbc.Alert(
                f"Error loading sample data: {str(e)}",
                color="danger",
                duration=5000,
                is_open=True,
            )
    return False, None

# Add callback to update charts periodically
@app.callback(
    [Output('stock-price-chart', 'figure'),
     Output('accuracy-chart', 'figure'),
     Output('predictions-table', 'children')],
    [Input('stock-selector', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_all_components(selected_stock, n_intervals):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    predictions = db_handler.get_predictions(
        stock_name=selected_stock,
        start_date=start_date,
        end_date=end_date
    )
    
    # Generate stock price chart
    chart = create_stock_chart(predictions, selected_stock)
    
    # Generate accuracy chart
    accuracy = create_prediction_accuracy_chart(predictions)
    
    # Generate predictions table
    headers = [
        html.Thead(html.Tr([
            html.Th("Date", style={"padding": "8px 16px", "text-align": "left"}),
            html.Th("Predicted", style={"padding": "8px 16px", "text-align": "left"}),
            html.Th("Actual", style={"padding": "8px 16px", "text-align": "left"}),
            html.Th("Accuracy", style={"padding": "8px 16px", "text-align": "left"})
        ], style={"background-color": "#f8f9fa"}))
    ]
    
    rows = []
    if predictions:
        for pred in predictions:
            accuracy_val = abs(pred['predictedPrice'] - pred['actualPrice']) / pred['actualPrice'] * 100
            rows.append(html.Tr([
                html.Td(pred['predictionDate'], style={"padding": "8px 16px"}),
                html.Td(f"${pred['predictedPrice']:.2f}", style={"padding": "8px 16px"}),
                html.Td(f"${pred['actualPrice']:.2f}", style={"padding": "8px 16px"}),
                html.Td(f"{accuracy_val:.1f}%", style={"padding": "8px 16px"})
            ], style={"border-bottom": "1px solid #dee2e6"}))
    
    table = dbc.Table(headers + [html.Tbody(rows)], bordered=False, hover=True, style={"margin-top": "0"})
    return chart, accuracy, table

# Update chat callback for more contextual responses
@app.callback(
    [Output('chat-history', 'children'),
     Output('chat-history-store', 'data')],
    [Input('send-chat', 'n_clicks'),
     Input('chat-input', 'n_submit')],
    [State('chat-input', 'value'),
     State('chat-history-store', 'data'),
     State('stock-selector', 'value')]
)
def update_chat(n_clicks, n_submit, input_text, chat_history, selected_stock):
    if (n_clicks is None and n_submit == 0) or not input_text:
        return html.Div(), chat_history or []
    
    try:
        # Get current predictions for context
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        predictions = db_handler.get_predictions(
            stock_name=selected_stock,
            start_date=start_date,
            end_date=end_date
        )
        
        # Extract stock symbol from input if mentioned
        input_lower = input_text.lower()
        stock_match = next((s for s in load_stock_list() if s.lower() in input_lower), selected_stock)
        
        # Get predictions for the mentioned stock
        if stock_match != selected_stock:
            predictions = db_handler.get_predictions(
                stock_name=stock_match,
                start_date=start_date,
                end_date=end_date
            )
        
        # Generate response based on context and question type
        if predictions:
            latest_pred = predictions[-1]
            prev_pred = predictions[-2] if len(predictions) > 1 else None
            
            price_trend = "up" if latest_pred['predictedPrice'] > latest_pred['actualPrice'] else "down"
            confidence = latest_pred['confidence']
            
            # Calculate price change percentage
            price_change = ((latest_pred['predictedPrice'] - latest_pred['actualPrice']) 
                          / latest_pred['actualPrice'] * 100)
            
            response = ""
            if 'why' in input_lower or 'reason' in input_lower:
                sentiment = latest_pred.get('metadata', {}).get('market_sentiment', 'neutral')
                trend_strength = "strong" if abs(price_change) > 2 else "moderate"
                response = (
                    f"For {stock_match}, I predict a {trend_strength} {price_trend}ward trend "
                    f"({abs(price_change):.1f}% change) with {confidence:.1f}% confidence. "
                    f"This is based on {sentiment} market sentiment and recent price movements. "
                )
                if prev_pred:
                    momentum = "increasing" if latest_pred['predictedPrice'] > prev_pred['predictedPrice'] else "decreasing"
                    response += f"The price momentum is {momentum}."
                    
            elif 'when' in input_lower:
                response = (
                    f"Based on my analysis of {stock_match}, I expect the {price_trend}ward movement "
                    f"to begin in the next trading day. The prediction has {confidence:.1f}% confidence "
                    f"and suggests a potential {abs(price_change):.1f}% change."
                )
            else:
                response = (
                    f"For {stock_match}, I predict the price will go {price_trend} by {abs(price_change):.1f}% "
                    f"with {confidence:.1f}% confidence. The current price is ${latest_pred['actualPrice']:.2f}."
                )
        else:
            response = f"I don't have enough recent data for {stock_match} to make a prediction. Try loading sample data first."
        
        # Update chat history
        chat_history = chat_history or []
        chat_history.append({
            "user": input_text,
            "bot": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Create chat display with improved styling
        chat_display = []
        for msg in chat_history:
            chat_display.extend([
                html.Div([
                    html.P(f"You: {msg['user']}", 
                          style={
                              "color": "#666",
                              "margin": "0",
                              "padding": "8px 12px",
                              "backgroundColor": "#e9ecef",
                              "borderRadius": "15px 15px 0 15px"
                          }),
                ], className="mb-2"),
                html.Div([
                    html.P(f"AI: {msg['bot']}", 
                          style={
                              "color": "#007bff",
                              "margin": "0",
                              "padding": "8px 12px",
                              "backgroundColor": "#007bff15",
                              "borderRadius": "15px 15px 15px 0"
                          }),
                ], className="mb-3")
            ])
        
        return html.Div(chat_display), chat_history
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return html.Div([
            html.P(f"Error: {str(e)}", style={"color": "red"})
        ]), chat_history or []

# Add callback to clear chat input after sending
@app.callback(
    Output('chat-input', 'value'),
    [Input('send-chat', 'n_clicks'),
     Input('chat-input', 'n_submit')]
)
def clear_chat_input(n_clicks, n_submit):
    return ''

if __name__ == '__main__':
    print("\nStarting dashboard server...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    
    app.run_server(
        host='127.0.0.1',
        port=8050,
        debug=False  # Disable debug mode temporarily
    )