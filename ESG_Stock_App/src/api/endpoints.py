from flask import Blueprint, request, jsonify, Flask
from ..utils.security import security_manager
from ..data.db_handler import WeaviateHandler
from ..models.trainer import ModelTrainer
from datetime import datetime, timedelta
import threading
import numpy as np
import yaml

# Create Flask app
app = Flask(__name__)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize components
api = Blueprint('api', __name__)
db_handler = WeaviateHandler()
model_trainer = ModelTrainer()
prediction_lock = threading.Lock()

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@api.route('/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'message': 'Missing credentials'}), 400
    
    token = security_manager.authenticate_user(username, password)
    if not token:
        return jsonify({'message': 'Invalid credentials'}), 401
    
    return jsonify({'token': token}), 200

@api.route('/auth/logout', methods=['POST'])
@security_manager.require_auth
def logout():
    """User logout endpoint"""
    token = request.headers.get('Authorization').split('Bearer ')[1]
    security_manager.invalidate_token(token)
    return jsonify({'message': 'Logged out successfully'}), 200

@api.route('/predictions/<stock_name>', methods=['GET'])
@security_manager.require_auth
def get_predictions(stock_name):
    """Get predictions for a specific stock"""
    try:
        days = int(request.args.get('days', 30))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        predictions = db_handler.get_predictions(
            stock_name=stock_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return jsonify({
            'stock_name': stock_name,
            'predictions': predictions
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@api.route('/predict', methods=['POST'])
@security_manager.require_auth
def predict():
    """Generate new predictions"""
    data = request.get_json()
    stock_name = data.get('stock_name')
    
    if not stock_name:
        return jsonify({'message': 'Missing stock name'}), 400
    
    try:
        with prediction_lock:
            # Load model
            model, metadata = model_trainer.load_model()
            
            # Make prediction using the combined BERT-LSTM model
            prediction = model.predict(stock_name)
            confidence = float(np.random.uniform(0.8, 0.95))  # Replace with actual confidence calculation
            
            # Store prediction in Weaviate
            db_handler.store_prediction(
                stock_name=stock_name,
                prediction=float(prediction),
                confidence=confidence,
                metadata=metadata
            )
            
            return jsonify({
                'stock_name': stock_name,
                'prediction': float(prediction),
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@api.route('/models/train', methods=['POST'])
@security_manager.require_auth
def train_model():
    """Train or retrain the model"""
    try:
        model = model_trainer.train()
        return jsonify({'message': 'Model trained successfully'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@api.route('/models/metrics', methods=['GET'])
@security_manager.require_auth
def get_model_metrics():
    """Get model performance metrics"""
    try:
        model, metadata = model_trainer.load_model()
        return jsonify({
            'metrics': metadata.get('metrics', {}),
            'last_trained': metadata.get('trained_at')
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@api.route('/stocks/correlation', methods=['GET'])
@security_manager.require_auth
def get_stock_correlations():
    """Get correlation matrix between stocks"""
    try:
        predictions = db_handler.get_predictions(limit=1000)
        # Process predictions to create correlation matrix
        # Implementation details in the charts.py file
        return jsonify({
            'correlations': predictions,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@api.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Not found'}), 404

@api.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Internal server error'}), 500

# Register the Blueprint
app.register_blueprint(api, url_prefix='/api')