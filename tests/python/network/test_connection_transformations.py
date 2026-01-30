import unittest, sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import connection_transformations as bpnct
import torch

class TestIndexBasedSplitter(unittest.TestCase):
    
    def test_case_one(self):
        """This test case tests whether the index-based splitter can split off 4 out of 7 dimensions along axis 2 of
        a tensor of shape [batch_size = 2, 2, 7].
        """
        
        # Create instance
        splitter = bpnct.IndexBasedSplitter(
            input_flatten_axes = (2,2),
            output_indices = [0,2,4,6],
            output_flatten_axes = (2,3),
            output_shape = [2,4,1]
        )

        # Create inputs
        # Shape = [2,2,7]
        x = torch.Tensor([
            [[  0., 100.,   1., 101.,   2., 102.,   3.], [  4., 103.,   5., 104.,   6., 105.,   7.]], # First instance
            [[  8., 106.,   9., 107.,  10., 108.,  11.], [ 12., 109.,  13., 110.,  14., 111.,  15.]]]) # Second instance
        

        # Process
        y_hat = splitter.forward(x=x)

        # Evaluate
        # First input, Shape = [2,2,4,1]
        y = torch.Tensor([[[[0],[1],[2],[3]],[[4],[5],[6],[7]]], # Instance 1
                             [[[8],[9],[10],[11]],[[12],[13],[14],[15]]]]) # Instance 2

        self.assertSequenceEqual(y.shape, y_hat.shape)
        assert all(torch.flatten(y)==torch.flatten(y_hat))
