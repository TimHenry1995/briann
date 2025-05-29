
import unittest
import sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import components as bnc
import torch
import os

class TestBrIANN__init__(unittest.TestCase):

    def test_areas(self):
        # Create an instance of BrIANN
        
        b = bnc.BrIANN(batch_size=4, configuration_file_path=os.path.join("tests","briann 1.json"))
        
        # Check if the index is a non-zero integer
        #self.assertIsInstance(time_frame.index, int, "Index should be an integer")
        #self.assertGreaterEqual(time_frame.index, 0, "Index should not be negative")



if __name__ == "__main__":
    #unittest.main()
    TestBrIANN__init__().test_areas()