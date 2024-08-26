import unittest
from models.predictive_model import build_lstm


class TestModels(unittest.TestCase):
    def test_lstm_model(self):
        model = build_lstm((10, 1))
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
