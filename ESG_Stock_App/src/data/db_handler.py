import weaviate
from weaviate.collections import Collection
import numpy as np
from typing import Dict, List, Optional
import yaml
import threading
from datetime import datetime, timedelta

class WeaviateHandler:
    def __init__(self, config_path: str = 'config.yaml'):
        self.predictions = []  # In-memory storage
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Weaviate client with v4 API
        self.client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams.from_url(
                url=self.config['weaviate']['url'],
                grpc_port=50051  # Default gRPC port for Weaviate
            ),
            additional_headers={
                "X-OpenAI-Api-Key": self.config['weaviate'].get('api_key', '')
            } if self.config['weaviate'].get('api_key') else None
        )
        self.lock = threading.Lock()
        self._setup_schema()
    
    def _setup_schema(self):
        """Sets up the Weaviate schema for stock predictions"""
        try:
            collection = self.client.collections.get("StockPrediction")
        except weaviate.exceptions.WeaviateEntityDoesNotExist:
            # Create collection if it doesn't exist
            collection = self.client.collections.create(
                name="StockPrediction",
                description="Stock price predictions with metadata",
                vectorizer_config=weaviate.config.Configure.Vectorizer.text2vec_contextionary(),
                properties=[
                    weaviate.config.Property(name="stockName", data_type=weaviate.config.DataType.TEXT),
                    weaviate.config.Property(name="predictionDate", data_type=weaviate.config.DataType.DATE),
                    weaviate.config.Property(name="predictedPrice", data_type=weaviate.config.DataType.NUMBER),
                    weaviate.config.Property(name="actualPrice", data_type=weaviate.config.DataType.NUMBER),
                    weaviate.config.Property(name="confidence", data_type=weaviate.config.DataType.NUMBER),
                    weaviate.config.Property(name="modelMetadata", data_type=weaviate.config.DataType.TEXT)
                ]
            )
    
    def store_prediction(self, 
                        stock_name: str,
                        prediction: float,
                        confidence: float,
                        actual_price: Optional[float] = None,
                        metadata: Optional[Dict] = None):
        """Store a stock prediction in Weaviate and in memory"""
        try:
            self.predictions.append({
                "stockName": stock_name,
                "predictionDate": datetime.now().isoformat(),
                "predictedPrice": prediction,
                "actualPrice": actual_price if actual_price is not None else prediction,
                "confidence": confidence,
                "modelMetadata": str(metadata or {})
            })
            print(f"Stored prediction for {stock_name}: {prediction}")
            
            collection = self.client.collections.get("StockPrediction")
            
            properties = {
                "stockName": stock_name,
                "predictionDate": datetime.now().isoformat(),
                "predictedPrice": prediction,
                "confidence": confidence,
                "modelMetadata": str(metadata or {})
            }
            
            if actual_price is not None:
                properties["actualPrice"] = actual_price
            
            collection.data.insert(properties)
            
        except Exception as e:
            print(f"Error storing prediction: {e}")
    
    def get_predictions(self, 
                       stock_name: Optional[str] = None,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       limit: int = 100) -> List[Dict]:
        """Retrieve predictions from Weaviate and memory with optional filtering"""
        try:
            filtered_predictions = self.predictions
            
            if stock_name:
                filtered_predictions = [p for p in filtered_predictions if p["stockName"] == stock_name]
            
            if start_date and end_date:
                filtered_predictions = [
                    p for p in filtered_predictions 
                    if start_date.isoformat() <= p["predictionDate"] <= end_date.isoformat()
                ]
            
            return filtered_predictions[-limit:]
            
        except Exception as e:
            print(f"Error retrieving predictions: {e}")
            return []
    
    def get_similar_predictions(self, 
                              prediction_vector: np.ndarray,
                              limit: int = 10) -> List[Dict]:
        """Find similar predictions using vector similarity search"""
        try:
            collection = self.client.collections.get("StockPrediction")
            query = collection.query.near_vector(
                content=prediction_vector.tolist(),
                limit=limit
            ).with_additional(["distance"])
            
            response = query.do()
            return [
                {**obj.properties, "distance": obj.metadata.distance}
                for obj in response.objects
            ]
            
        except Exception as e:
            print(f"Error performing similarity search: {e}")
            return []
    
    def cleanup_old_predictions(self, days_old: int = 30):
        """Remove predictions older than specified days"""
        try:
            collection = self.client.collections.get("StockPrediction")
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            collection.data.delete_many(
                weaviate.query.Filter.by_property("predictionDate").less_than(cutoff_date.isoformat())
            )
            
        except Exception as e:
            print(f"Error cleaning up old predictions: {e}")