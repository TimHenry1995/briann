import unittest
from pybriann.network.components import TimeFrame
import torch

class TestTimeFrame(unittest.TestCase):
    example_tensor = torch.Tensor([1, 2, 3, 4, 5])

    def test_index_is_non_zero_int(self):
        # Create an instance of TimeFrame with a sample index
        time_frame = TimeFrame(state=TestTimeFrame.example_tensor, index=5, start_time=0.0, duration=10.0)
        
        # Check if the index is a non-zero integer
        self.assertIsInstance(time_frame.index, int, "Index should be an integer")
        self.assertGreaterEqual(time_frame.index, 0, "Index should not be negative")

if __name__ == "__main__":
    unittest.main()