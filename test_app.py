"""
Automated Test Suite for HomeValue AI FastAPI Backend
"""

import os
import sys
import unittest

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi.testclient import TestClient

# Ensure app can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import app

client = TestClient(app)

class TestHousePriceApp(unittest.TestCase):

    def test_health_endpoint(self):
        """Test GET /health returns status ok."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        print("✓ GET /health passed")

    def test_metrics_endpoint(self):
        """Test GET /metrics returns valid model metrics and features."""
        response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("linear", data["models"])
        self.assertIn("random_forest", data["models"])
        self.assertIn("top_features", data)
        self.assertIn("actual_vs_predicted", data)
        self.assertEqual(len(data["top_features"]), 5)
        self.assertEqual(len(data["actual_vs_predicted"]), 100)
        print("✓ GET /metrics passed")

    def test_predict_random_forest(self):
        """Test POST /predict with Random Forest model."""
        payload = {
            "living_area": 1710,
            "overall_quality": 7,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2003,
            "garage_cars": 2,
            "basement_area": 856,
            "neighborhood": "CollgCr",
            "model": "random_forest"
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_price", data)
        self.assertIn("range_low", data)
        self.assertIn("range_high", data)
        self.assertEqual(data["model_used"], "Random Forest Regressor")
        self.assertGreater(data["predicted_price"], 100000)
        # Check +-8% range formula
        expected_low = round(data["predicted_price"] * 0.92, 2)
        expected_high = round(data["predicted_price"] * 1.08, 2)
        self.assertEqual(data["range_low"], expected_low)
        self.assertEqual(data["range_high"], expected_high)
        print(f"✓ POST /predict (Random Forest) passed -> Predicted: ${data['predicted_price']:,.2f}")

    def test_predict_linear(self):
        """Test POST /predict with Linear Regression model."""
        payload = {
            "living_area": 1710,
            "overall_quality": 7,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2003,
            "garage_cars": 2,
            "basement_area": 856,
            "neighborhood": "CollgCr",
            "model": "linear"
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_price", data)
        self.assertEqual(data["model_used"], "Linear Regression")
        print(f"✓ POST /predict (Linear Regression) passed -> Predicted: ${data['predicted_price']:,.2f}")

    def test_predict_validation_errors(self):
        """Test input validation for invalid fields."""
        # Invalid overall_quality (> 10)
        invalid_payload = {
            "living_area": 1500,
            "overall_quality": 15,
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 2000,
            "garage_cars": 2,
            "basement_area": 500,
            "neighborhood": "CollgCr",
            "model": "random_forest"
        }
        res = client.post("/predict", json=invalid_payload)
        self.assertEqual(res.status_code, 422)

        # Invalid living_area (<= 0)
        invalid_payload["overall_quality"] = 5
        invalid_payload["living_area"] = -100
        res = client.post("/predict", json=invalid_payload)
        self.assertEqual(res.status_code, 422)
        print("✓ Input validation (Pydantic 422 errors) passed")

    def test_frontend_serving(self):
        """Test that root / serves index.html."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HomeValue AI", response.text)
        self.assertIn("predict-form", response.text)
        print("✓ Frontend static serving (index.html) passed")

if __name__ == "__main__":
    unittest.main()
