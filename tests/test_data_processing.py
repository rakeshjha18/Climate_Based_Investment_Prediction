import unittest
from scripts.preprocess_data import preprocess_data
import os


class TestDataProcessing(unittest.TestCase):
    def test_preprocess_data(self):
        preprocess_data("../data/raw/environmental_data.csv", "../data/processed/test_processed_data.csv")
        self.assertTrue(os.path.exists("../data/processed/test_processed_data.csv"))


if __name__ == "__main__":
    unittest.main()
