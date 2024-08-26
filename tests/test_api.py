import unittest
from api.app import app


class TestAPI(unittest.TestCase):
    def test_predict_endpoint(self):
        with app.test_client() as client:
            response = client.post('/predict', json=[[1, 2, 3, 4, 5]])
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
