import sys, os
sys.path.append(os.path.abspath(""))
from briann.utilities import core as bpuc
import torch
from typing import Tuple, List, Dict

class Splitter(bpuc.Adapter):
    """This class is to be used in the :py:meth:`~briann.network.core.Connection.forward` method. It takes an input x that is the current :py:meth:`~briann.network.core.TimeFrame.state` from the sending 
    :py:class:`~briann.network.core.Area.output_time_frame_accumulator` and then splits off the part that is relevant to the calling :py:class:`~briann.network.core.Connection` object."""
    
    def __init__(self) -> "Splitter":
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class IndexBasedSplitter(torch.nn.Module):
    """The IndexBasedSplitter maps its input to its output by using indices in the :py:meth:`~briann.network.core.IndexBasedSplitter.forward`. To configure this mapping,
    the following arguments are used.

    :param input_flatten_axes: Sets the :py:meth:`~briann.network.core.IndexBasedSplitter.input_flatten_axes` of this object.
    :type input_flatten_axes: Tuple[int,int]
    :param input_indices: Sets the :py:meth:`~briann.network.core.IndexBasedSplitter.input_indices` of this object.
    :type input_indices: List[int]
    :param output_flatten_axes: Sets the :py:meth:`~briann.network.core.IndexBasedSplitter.output_flatten_axes` of this object.
    :type output_flatten_axes: Tuple[int,int]
    :param output_shape: Sets the :py:meth:`~briann.network.core.IndexBasedSplitter.output_shape` of this object.
    :type output_shape: List[int]
    """
    
    def __init__(self, input_flatten_axes: Tuple[int,int],
                 input_indices: List[int],
                 output_flatten_axes: Tuple[int,int],
                 output_shape: List[int]) -> "IndexBasedSplitter":
        
        # Super
        super().__init__()
        
        # Properties
        self.input_flatten_axes = input_flatten_axes
        self.input_indices = input_indices
        self.output_flatten_axes = output_flatten_axes
        self.output_shape = output_shape

    @property
    def input_flatten_axes(self) -> Tuple[int,int]:
        """:return: The axes along which the input will be flattened inside :py:meth:`~briann.network.core.IndexBasedSplitter.forward` before selecting entries from it. This is a Tuple of two ints, where the first int is the axis at which flattening starts and the second int is the axis (inclusive) at which it will end. 
        :rtype: Tuple[int, int]"""

        return self._input_flatten_axes
    
    @input_flatten_axes.setter
    def input_flatten_axes(self, new_value: Tuple[int,int]) -> None:
        # Input validity
        # - Type
        if not isinstance(new_value, Tuple) or not len(new_value) == 2 or not all([isinstance(entry, int) for entry in new_value]): raise TypeError(f"The input_flatten_axes of a IndexBasedSplitter are expected to be of type Tuple[int,int] but were {new_value}.")

        # - Value
        if not all([entry >= 1 for entry in new_value]): raise ValueError(f"When setting the input_flatten_axes, no value below 1 is permitted, yet {new_value} was provided.")

        # Set property
        self._input_flatten_axes = new_value

    @property
    def input_indices(self) -> Dict[int, List[int]]:
        """:return: A list of indices that specify which entries of the flattened input :py:class:`torch.Tensor` shall be moved to in the flattened output tensor. The length of this list must factor into :py:meth:`~briann.network.core.IndexBasedSplitter.output_shape` along the :py:meth:`~briann.network.core.IndexBasedSplitter.output_flatten_axes`.
        :rtype: List[int]"""
        return self._input_indices
    
    @input_indices.setter
    def input_indices(self, new_value: List[int]) -> None:

        # Input validity
        # - Type
        if not all([isinstance(entry, int) for entry in new_value]): raise TypeError(f"The output_indices of an IndexBasedMerger was expected to be a List[int], but was tried to be set to {new_value}")

        # - Values
        for tensor_index in new_value:
            if tensor_index < 0: raise ValueError(f"An illegal index ({tensor_index}) was tried to be set as tensor_index when setting the output_indices of an IndexBasedSplitter.")

        # Set value
        self._input_indices = new_value

    @property
    def output_flatten_axes(self) -> Tuple[int,int]:
        """:return: A Tuple of two axes, namely the start_axis and end_axis (inclusive). Axes spanned by the start_axis and end_axis are the axes along which the output tensor will be flattened before its entries are filled with the corresponing ones from the input tensor. The calculation of these axes DOES include the initial batch axis and is thus assumed to lead to axes greater than 0.
        :rtype: Tuple[int,int]"""
        return self._output_flatten_axes
    
    @output_flatten_axes.setter
    def output_flatten_axes(self, new_value: Tuple[int,int]) -> None:

        # Input validity
        # - Type
        if not len(new_value) == 2 or not all([isinstance(entry, int) for entry in new_value]): raise TypeError(f"The output_flatten_axes of an IndexBasedSplitter was expected to be a Tuple[int,int], but was tried to be set to {new_value}.")

        # - Values
        for axis in new_value:
            if axis < 1: raise ValueError(f"An illegal axis ({axis}) was tried to be set as input_flatten_axes of an IndexBasedSplitter.")

        # Set value
        self._output_flatten_axes = new_value

    @property
    def output_shape(self) -> List[int]:
        """:return: The desired shape for the :py:class:`torch.Tensor` that the :py:meth:`~briann.network.core.IndexBasedSplitter.forward` method shall output.
        :rtype: List[int]"""
        return self._output_shape
    
    @output_shape.setter
    def output_shape(self, new_value: List[int]) -> None:

        # Input validity
        # - Type
        if not all([isinstance(entry, int) for entry in new_value]): raise TypeError(f"The output_shape of an IndexBasedSplitter was expected to be a List[int], but was tried to be set to {new_value}.")

        # - Values
        for dimensionality in new_value:
            if dimensionality < 0: raise ValueError(f"An illegal shape ({new_value}) was tried to be set for an IndexBasedSplitter.")

        # Set value
        self._output_shape = new_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method first flattens the input tensor `x` along :py:meth:`~briann.network.core.IndexBasedSplitter.input_flatten_axes`, then extracts the corresponding entries using 
        the indices from :py:meth:`~briann.network.core.IndexBasedSplitter.output_indices` and then unflattens the output along the axes specified in 
        :py:meth:`~briann.network.core.IndexBasedSplitter.output_flatten_axes` to arrive at the output shape specified in :py:meth:`~briann.network.core.IndexBasedSplitter.output_shape`."""

        # Flatten x
        x_axis = self.input_flatten_axes[0]
        x = torch.flatten(input=x, start_dim=x_axis, end_dim=self.input_flatten_axes[1])

        # Split by extracting relevant section
        indices = self.input_indices
        y = torch.index_select(input=x, dim=x_axis, index=torch.tensor(indices))

        # Unflatten y
        start_axis = self.output_flatten_axes[0]
        end_axis = self.output_flatten_axes[1] + 1 # +1 since the end_index should be inclusive
        shape_along_unflattened_axes = self.output_shape[start_axis-1:end_axis-1] # -1 since the provided output shape does not include the initial batch_size axis
        y = torch.unflatten(input=y, dim=start_axis, sizes=shape_along_unflattened_axes)

        # Output
        return y
