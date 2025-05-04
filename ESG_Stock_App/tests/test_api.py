import unittest
from src.api.endpoints import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'healthy')

    def test_prediction_endpoint(self):
        sample_data = {
            "stock_name": "AAPL"
        }
        response = self.app.post('/api/predict', json=sample_data)
        self.assertEqual(response.status_code, 401)  # Should fail without auth

    def test_get_predictions(self):
        response = self.app.get('/api/predictions/AAPL')
        self.assertEqual(response.status_code, 401)  # Should fail without auth

if __name__ == '__main__':
    unittest.main()