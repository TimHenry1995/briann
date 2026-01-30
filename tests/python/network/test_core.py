import unittest, sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import core as bpnc
import torch

class TestIndexBasedMerger(unittest.TestCase):
    
    def test_case_one(self):
        """In this test case, there are two inputs. The first is of shape [1,2,4,1] and the second of shape [1,2,2], hence
        a batch_size of 1.
        The first tensor will be flattened along its last two axes. The second tensor will stay as it is.
        The dimensions of the last axis of both tensors will be concatenated into the output tensor.
        The output tensor will not be reshaped.
        """
        
        # Create instance
        merger = bpnc.IndexBasedMerger(connection_index_to_input_flatten_axes={0:[2,3], 1:[2,2]},
                                       connection_index_to_output_indices={0:[0,1,2,3], 1:[4,5]},
                                       output_flatten_axes=[2,2],
                                       final_output_shape=[2,6]
        )

        # Create inputs
        # Batch_size is 1 here
        x = {
            # First input, Shape = [1,2,4,1]
            0: torch.Tensor([[[[0],[1],[2],[3]],[[4],[5],[6],[7]]]]), # First instance
            
            # Second input, Shape = [1,2,2]
            1: torch.Tensor([[[11,12], [13,14]]]) # First instance
        }

        # Process
        y_hat = merger.forward(x=x)

        # Evaluate
        y = torch.Tensor([[[ 0.,  1.,  2.,  3., 11., 12.], [ 4.,  5.,  6.,  7., 13., 14.]]]) # First instance
        self.assertSequenceEqual(y.shape, y_hat.shape)
        assert all(torch.flatten(y)==torch.flatten(y_hat))

    def test_case_two(self):
        """In this test case, there are two inputs. The first is of shape [2,2,4,1] and the second of shape [2,2,2], hence 
        a batch_size of 2.
        The first tensor will be flattened along its last two axes. The second tensor will stay as it is.
        The dimensions of the last axis of both tensors will be concatenated into the output tensor.
        The output tensor will not be reshaped.
        """

        # Create instance
        merger = bpnc.IndexBasedMerger(connection_index_to_input_flatten_axes={0:[2,3], 1:[2,2]},
                                       connection_index_to_output_indices={0:[0,1,2,3], 1:[4,5]},
                                       output_flatten_axes=[2,2],
                                       final_output_shape=[2,6]
        )

        # Create inputs
        # Batch_size is 1 here
        x = {
            # First input, Shape = [2,2,4,1]
            0: torch.Tensor([[[[0],[1],[2],[3]],[[4],[5],[6],[7]]], # First instance
                             [[[10],[11],[12],[13]],[[14],[15],[16],[17]]]]), # Second instance
            # Second input, Shape = [2,2,2]
            1: torch.Tensor([[[11,12],[13,14]], # First instance
                             [[111,112],[113,114]]]) # Second instance
        }

        # Process
        y_hat = merger.forward(x=x)

        # Evaluate
        y = torch.Tensor([[[ 0.,  1.,  2.,  3., 11., 12.], [ 4.,  5.,  6.,  7., 13., 14.]], # First instance
                        [[ 10.,  11.,  12.,  13., 111., 112.], [ 14.,  15.,  16.,  17., 113., 114.]]]) # Second instance
        self.assertSequenceEqual(y.shape, y_hat.shape)
        assert all(torch.flatten(y)==torch.flatten(y_hat))

    def test_case_three(self):
        """In this test case, there are two inputs. The first is of shape [2,2,4,1] and the second of shape [2,2,2], hence 
        a batch_size of 2.
        The first tensor will be flattened along its last two axes. The second tensor will stay as it is.
        The dimensions of the last axis of both tensors will be interleaved into the output tensor.
        The output tensor will not be reshaped.
        """

        # Create instance
        merger = bpnc.IndexBasedMerger(connection_index_to_input_flatten_axes={0:[2,3], 1:[2,2]},
                                       connection_index_to_output_indices={0:[0,2,4,6], 1:[1,3,5]},
                                       output_flatten_axes=[2,2],
                                       final_output_shape=[2,7]
        )

        # Create inputs
        # Batch_size is 1 here
        x = {
            # First input, Shape = [2,2,4,1]
            0: torch.Tensor([[[[0],[1],[2],[3]],[[4],[5],[6],[7]]], # Instance 1
                             [[[8],[9],[10],[11]],[[12],[13],[14],[15]]]]), # Instance 2
            # Second input, Shape = [2,2,3]
            1: torch.Tensor([[[100,101,102],[103,104,105]], # Instance 1
                             [[106,107,108],[109,110,111]]]) # Instance 2
        }

        # Process
        y_hat = merger.forward(x=x)

        # Evaluate
        y = torch.Tensor([
        [[  0., 100.,   1., 101.,   2., 102.,   3.], [  4., 103.,   5., 104.,   6., 105.,   7.]], # First instance
        [[  8., 106.,   9., 107.,  10., 108.,  11.], [ 12., 109.,  13., 110.,  14., 111.,  15.]]]) # Second instance
        
        self.assertSequenceEqual(y.shape, y_hat.shape)
        assert all(torch.flatten(y)==torch.flatten(y_hat))

if __name__ == '__main__':
    unittest.main()