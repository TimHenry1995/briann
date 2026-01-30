"This module collects all necessary components to build a BrIANN model."
import torch
from typing import List, Dict, Deque, Set, Any, Tuple
from collections import deque

import sys, os
sys.path.append(os.path.abspath(""))

from briann.utilities import callbacks as bpuc
from briann.utilities import core as bpuco

import networkx as nx
from abc import ABC, abstractmethod

class TimeFrame():
    """A time-frame in the simulation that holds a temporary state of an :py:class:`~briann.network.core.Area`. 

    :param state: Sets the :py:attr:`~briann.network.core.TimeFrame.state` of this time frame.
    :type state: :py:class:`torch.Tensor`
    :param time_point: Sets the :py:attr:`~briann.network.core.TimeFrame.time_point` of this time frame.
    :type time_point: float
    """
    
    def __init__(self, state: torch.Tensor, time_point: float) -> None:
        
        # Set properties
        self.state = state
        self.time_point = time_point    

    @property
    def state(self) -> torch.Tensor:
        """:return: The state of the time frame. This is a :py:class:`torch.tensor`, for instance of shape [instance count, dimensionality].
        :rtype: torch.Tensor"""
        return self._state

    @state.setter
    def state(self, new_value: torch.Tensor) -> None:
        
        # Check input validity
        if not isinstance(new_value, torch.Tensor):
            raise TypeError(f"The state must be a torch.Tensor but was {type(new_value)}.")
        self._state = new_value

    @property
    def time_point(self) -> float:
        """:return: The time point at which this time frame's :py:meth:`~briann.network.core.TimeFrame.state` occured.
        :rtype: float"""
        return self._time_point
    
    @time_point.setter
    def time_point(self, new_value: float | int) -> None:
        
        # Check input validity
        if isinstance(new_value, int): new_value = (float)(new_value)
        if not isinstance(new_value, float):
            raise TypeError(f"The time_point must be a float but was {type(new_value)}.")
        
        # Set property
        self._time_point = new_value

    def __repr__(self) -> str:
        return f"TimeFrame(time_point={self.time_point}, state shape={self.state.shape})"

class TimeFrameAccumulator():
    """This class is used to accumulate :py:class:`.TimeFrame` objects. Accumulation happens by adding new time-frames into the accumulator's
    own time-frame using the :py:meth:`~briann.network.core.TimeFrameAccumulator.accumulate` function. An important feature of the accumulator is that during
    every update, the currently stored information decays according to the provided `decay_rate` and the time since the last update. 
    This is done to ensure that older information has less influence on the current state of the accumulator than new information.

    :param initial_time_frame: Sets the :py:attr:`~briann.network.core.TimeFrameAccumulator.initial_time_frame` and :py:attr:`~briann.network.core.TimeFrameAccumulator.time_frame` of this time frame accumulator.
    :type initial_time_frame: :py:class:`~briann.network.core.TimeFrame`
    :param decay_rate: Sets the :py:meth:`~briann.network.core.TimeFrameAccumulator.decay_rate` property of self.
    :type decay_rate: float
    """

    def __init__(self, initial_time_frame: TimeFrame, decay_rate: float) -> None:
           
        # Set initial time-frame and time-frame
        if not isinstance(initial_time_frame, TimeFrame):
            raise TypeError(f"The initial_time_frame was expected to be a TimeFrame but was {type(initial_time_frame)}.")
        self._time_frame = initial_time_frame
        self._initial_time_frame = initial_time_frame

        # Set decay rate
        self.decay_rate = decay_rate
    
    @property
    def decay_rate(self) -> float:
        """:return: The rate taken from the interval [0,1] at which the energy of the :py:meth:`~briann.network.core.TimeFrame.state` of :py:meth:`~briann.network.core.TimeFrameAccumulator.time_frame` decays as time passes. This rate is recommended to be in the range (0,1), in order to have true exponential decay. If set to 1, there is no decay, if set to 0, there is no memory. See py:meth:`~.TimeFrameAccumulator.accumulate` for details.
        :rtype: float"""
        return self._decay_rate
        
    @decay_rate.setter
    def decay_rate(self, new_value: float) -> None:

        # Check input validity
        if not isinstance(new_value, float):
            raise TypeError(f"The decay_rate should be a float but was {type(new_value)}.")
        
        if new_value < 0 or new_value > 1: 
            raise ValueError(f"The decay_rate should not be outside the interval [0,1] but was set to {new_value}.")

        # Set property
        self._decay_rate = new_value

    def accumulate(self, time_frame: TimeFrame) -> None:
        """Sets the :py:meth:`~briann.network.core.TimeFrame.state` of the :py:meth:`~briann.network.core.TimeFrameAccumulator.time_frame` of self equal to the weighted sum of 
        the state of the new `time_frame` and the state the current time frame of self. The weight for the old state is 
        w = :py:meth:`~briann.network.core.TimeFrameAccumulator.decay_rate`^dt, where dt is the time of the provided `time_frame` minus the time-frame currently 
        held by self. The weight for the new `time_frame` is simply equal to 1.
        This method also sets the :py:meth:`~briann.network.core.TimeFrame.time_point` of the time-frame of self equal to that of the new `time_frame`.

        :param time_frame: The new time-frame to be added to the :py:meth:`~briann.network.core.TimeFrameAccumulator.time_frame` of self.
        :type time_frame: :py:class:`~briann.network.core.TimeFrame`
        :raises ValueError: If the state of `time_frame` does not have the same shape as that of the current time-frame of self.
        :raises ValueError: If the time-point of `time_frame` is earlier than that of the current time-frame of self.
        :return: None
        """
        
        # Ensure input validity
        if not isinstance(time_frame, TimeFrame):
            raise TypeError(f"The time_frame must be a TimeFrame but was {type(time_frame)}.")
        if not time_frame.state.shape == self._time_frame.state.shape:
            raise ValueError(f"The state of the new time_frame must have the same shape as that of self. Expected {self._time_frame.state.shape} but got {time_frame.state.shape}.")
        if time_frame.time_point < self._time_frame.time_point:
            raise ValueError("The new time_frame must not occur earlier in time than the current time-frame of self.")
        
        # Update time frame
        dt = time_frame.time_point - self._time_frame.time_point
        self._time_frame = TimeFrame(state=self._time_frame.state*self.decay_rate**dt + time_frame.state, time_point=time_frame.time_point)

    def time_frame(self, current_time: float) -> TimeFrame:
        """Provides a :py:class:`~briann.network.core.TimeFrame` that holds the time-discounted sum of all :py:class:`~briann.network.core.TimeFrame` objects added via the :py:meth:`~briann.network.core.TimeFrameAccumulator.accumulate` method.

        :param current_time: The current time, used to discount the state of self.
        :type current_time: float
        :raises ValueError: If `current_time` is earlier than the time-point of the current time-frame of self.
        :return: The time-discounted time-frame of this accumulator.
        :rtype: :py:class:`~briann.network.core.TimeFrame`
        """

        # Ensure data correctness
        if isinstance(current_time, int): current_time = (float)(current_time)
        if not isinstance(current_time, float):
            raise TypeError(f"The current_time must be a float but was {type(current_time)}.")
        if self._time_frame.time_point > current_time:
            raise ValueError(f"When reading a TimeFrame, the provided current_time ({current_time}) must be later than that of the time-frame held by self ({self._time_frame.value.time_point}).")
        
        # Update time frame
        dt = current_time - self._time_frame.time_point
        self._time_frame = TimeFrame(state=self._time_frame.state*self.decay_rate**dt, time_point=current_time)

        return self._time_frame 

    def reset(self, initial_time_frame: TimeFrame = None) -> None:
        """Resets the :py:meth:`~briann.network.core.TimeFrameAccumulator.time_frame` of self. If `initial_time_frame` is provided, then this one will
        be used for reset and saved in :py:meth:`~briann.network.core.TimeFrameAccumulator.initial_time_frame`. Otherwise, the one provided during construction will be used.

        :param initial_time_frame: The time-frame to be used to set :py:meth:`~briann.network.core.TimeFrameAccumulator.time_frame` and :py:meth:`~briann.network.core.TimeFrameAccumulator.initial_time_frame` of self.
        :type initial_time_frame: TimeFrame, optional, defaults to None.
        """

        if initial_time_frame != None:
            # Ensure input validity
            if not isinstance(initial_time_frame, TimeFrame):
                raise TypeError(f"The initial_time_frame must be a TimeFrame but was {type(initial_time_frame)}.")
            
            # Set properties
            self._time_frame = initial_time_frame
            self._initial_time_frame = initial_time_frame
        else:
            self._time_frame = self._initial_time_frame

    def __repr__(self) -> str:
        return f"TimeFrameAccumulator(decay_rate={self.decay_rate}, state shape={self._time_frame.state.shape}, time_point={self._time_frame.time_point})"
 
class Merger(bpuco.Adapter):
    """This class (and in particular its forward method) is to be used inside the :py:meth:`~briann.network.core.Area.collect_inputs` method to merge all the 
    collected inputs into one :py:class:`torch.Tensor`.  
    """
    
    @abstractmethod
    def forward(self, x: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        :param x: A dictionary of [int, tensor] where a key is an index of a connection and a value is a tensor to be processed. This input is assumed to be non-empty and an exception will be raised if the assumption is violated.
        :type x: Dict[int, torch.Tensor]
        :return: A single tensor that is the combination of the values of `x`.
        :rtype: :py:class:`torch.Tensor`
        """

class AdditiveMerger(Merger):
    """This merger adds all inputs."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Adds all tensors stored in the values of input dictionary `x`.

        :param x: A dictionary of [int, tensor] where a key is an index of a connection and a value is a tensor to be processed. This input is assumed to be non-empty and an exception will be raised if the assumption is violated.
        :type x: Dict[int, torch.Tensor]
        """

        # Input validity
        if not isinstance(x, Dict) or not all([isinstance(key, int) and isinstance(value, torch.Tensor) for key, value in x.items()]): 
            raise TypeError(f"The input x to a Merger's forward method should be of type Dict[int, torch.Tensor], not {type(x)}.")
        
        if len(x) == 0: raise ValueError(f"The input x to a Merger's forward method should not be empty.")

        # Merge
        xs = list(x.values())
        y = xs[0]
        for x_i in xs[1:]:
            y += x_i

        # Output
        return y

class IndexBasedMerger(Merger):
    """Maps the dimensions of the input tensors given to the :py:meth:`~briann.network.core.IndexBasedMerger.forward` method to the output tensor.
    
    :param connection_index_to_input_flatten_axes: Sets the :py:meth:`~briann.network.core.IndexBasedMerger.connection_index_to_input_flatten_axes` property of this instance.
    :type connection_index_to_input_flatten_axes: Dict[int, Tuple[int,int]]
    :param connection_index_to_output_indices: Sets the :py:meth:`~briann.network.core.IndexBasedMerger.connection_index_to_output_indices` property of this instance.
    :type connection_index_to_output_indices: Dict[int, Tuple[int,int]]
    :param output_flatten_axes: Sets the :py:meth:`~briann.network.core.IndexBasedMerger.output_flatten_axes` property of this instance.
    :type output_flatten_axes: Tuple[int,int]
    :param final_output_shape: Sets the :py:meth:`~briann.network.core.IndexBasedMerger.final_output_shape` property of this instance.
    """

    def __init__(self, 
                 connection_index_to_input_flatten_axes: Dict[int, Tuple[int,int]],
                 connection_index_to_output_indices: Dict[int, List[int]], 
                 output_flatten_axes: Tuple[int,int], 
                 final_output_shape: List[int]) -> None:
        
        # Super
        super().__init__()
        
        # Properties
        self._connection_index_to_input_flatten_axes = connection_index_to_input_flatten_axes
        self._connection_index_to_output_indices = connection_index_to_output_indices
        self._output_flatten_axes = output_flatten_axes
        self._final_output_shape = final_output_shape

    @property
    def connection_index_to_input_flatten_axes(self) -> Dict[int, Tuple[int,int]]:
        """:return:A dictionary mapping the :py:meth:`~briann.network.core.Connection.index` to a Tuple of two axes, namely the start_axis and end_axis (inclusive). Axes spanned by the start_axis and end_axis are the axes along which the given input tensor will be flattened before its dimensions are mapped onto the output tensor. The calculation of these axes DOES include the initial batch axis and is thus assumed to be greater than 0.
        :rtype: Dict[int, Tuple[int,int]]"""
        return self._connection_index_to_input_flatten_axes
    
    @connection_index_to_input_flatten_axes.setter
    def connection_index_to_input_flatten_axes(self, new_value: Dict[int, Tuple[int,int]]) -> None:

        # Input validity
        # - Type
        if not isinstance(new_value, Dict) or not all([isinstance(key, int) for key in new_value.keys()]) or not all([isinstance(value, Tuple) for value in new_value.values()]) or not all([len(value) == 2 for value in new_value.values()]) or not all([all([isinstance(entry, int) for entry in value]) for value in new_value.values()]): raise TypeError(f"The connection_index_to_input_flatten_axes of an IndexBasedMerger was expected to be a Dict[int, Tuple[int,int]], but was tried to be set to {new_value}")

        # - Values
        for connection_index, axes in new_value:
            if connection_index < 0: raise ValueError(f"An invalid connection index of {connection_index} < 0 was used when trying to set the connection_index_to_input_flatten_axes of an IndexBasedMerger.")
            for axis in axes:
                if axis < 1: raise ValueError(f"An illegal axis ({axis}) was tried to be set as flatten-axis when setting the connection_index_to_input_flatten_axes of an IndexBasedMerger.")

        # Set value
        self._connection_index_to_input_flatten_axes = new_value

    @property
    def connection_index_to_output_indices(self) -> Dict[int, List[int]]:
        """:return:A dictionary mapping the :py:meth:`~briann.network.core.Connection.index` to a list of indices. These latter indices specify where the dimensions of the connection's flattened input :py:class:`torch.Tensor` shall be moved to in the flattened output tensor. Important, the tensor_indices must all be unique and, when joined and sorted, give a contiguous list starting at 0. The length of this list must factor into the :py:meth:`~briann.network.core.IndexBasedMerger.final_output_shape` along the :py:meth:`~briann.network.core.IndexBasedMerger.output_flatten_axes`.
        :rtype: Dict[int, List[int]]"""
        return self._connection_index_to_output_indices
    
    @connection_index_to_output_indices.setter
    def connection_index_to_output_indices(self, new_value: Dict[int, List[int]]) -> None:

        # Input validity
        # - Type
        if not isinstance(new_value, Dict) or not all([isinstance(key, int) for key in new_value.keys()]) or not all([isinstance(value, List) for value in new_value.values()]) or not all([all([isinstance(entry, int) for entry in value]) for value in new_value.values()]): raise TypeError(f"The connection_index_to_output_indices of an IndexBasedMerger was expected to be a Dict[int, List[int]], but was tried to be set to {new_value}")

        # - Values
        for connection_index, tensor_indices in new_value:
            if connection_index < 0: raise ValueError(f"An invalid connection index of {connection_index} < 0 was used when trying to set the connection_index_to_output_indices of an IndexBasedMerger.")
            for tensor_index in tensor_indices:
                if tensor_index < 0: raise ValueError(f"An illegal index ({tensor_index}) was tried to be set as tensor_index when setting the connection_index_to_output_indices of an IndexBasedMerger.")

        # Set value
        self._connection_index_to_output_indices = new_value

    @property
    def output_flatten_axes(self) -> Tuple[int,int]:
        """:return: The first axis and last axis (inclusive) along which the output tensor shall initially be flattened while mapping the inputs onto it. The calculation of these axes DOES include the initial batch axis and is thus assumed to be greater than 0.
        :rtype: Tuple[int,int]"""
    
        return self._output_flatten_axes
    
    @output_flatten_axes.setter
    def output_flatten_axes(self, new_value: Tuple[int,int]) -> None:
        # Input validity
        # - Type
        if not isinstance(new_value, Tuple) or not len(new_value) == 2 or not all([isinstance(entry, int) for entry in new_value]): raise TypeError(f"The output_flatten_axes of an IndexBasedMerger was expected to be a Tuple[int,int], but was tried to be set to {new_value}")

        # - Values
        for axis in new_value:
            if axis < 1: raise ValueError(f"An illegal axis ({axis}) was tried to be set as output flatten-axis when setting the output_flatten_axes of an IndexBasedMerger.")

        # Set property
        self._output_flatten_axes = new_value
        
    @property
    def final_output_shape(self) -> List[int]:
        """:return: The final output shape after unflattening the output tensor along the specified `output_flatten_axes`. This shape does NOT include the initial batch axis, acknowledgeing that the batch-size is not necssarily known during configuration of this object.
            :rtype: List[int]"""
        
        return self._final_output_shape
    
    @final_output_shape.setter
    def final_output_shape(self, new_value: List[int]) -> None:

        # Input validity
        # - Type
        if not isinstance(new_value, List) or not all([isinstance(entry, int) for entry in new_value]): raise TypeError(f"The final_output_shape of an IndexBasedMerger was expected to be a List[int], but was tried to be set to {new_value}")

        # - Values
        for entry in new_value:
            if entry < 0: raise ValueError(f"An illegal index ({entry}) was tried to be set as final_output_shape of an IndexBasedMerger.")

        # Set property
        self._output_flatten_axes = new_value

    def forward(self, x: Dict[int, torch.Tensor]) -> torch.Tensor:
        """For each input tensor in `x`, this method first flattens the input along the axes specified during initialization and then moves the dimensions in their 
        existing order to the indices of the output tensor that were specified during initialization. At this point, the output tensor is flat along the 
        axes specified during initialization. Therafter, the output tensor is reshaped along those axes to reach its final output shape specified during initialization. 
        It is thus assumed that the input tensors and the output tensor all have the same shape along the remaining axes.
        
        For example, assume the inputs have shapes
        
        - first input:  [batch_size, 2, 4, 5] which will be flattened to [batch_size, 2, 20] 
        - second input: [batch_size, 2, 10]   which will be "flattened" to [batch_size, 2, 10]. 
        
        Then, assume the final output shape is [batch_size] + [2,10,3] which will be flattened to [batch_size] + [2,30]. 

        The mapping is performed on the flattened tensors. For simplicity, say the 20 dimensions of the first input's axis 2 are mapped to the first 20 dimensions of 
        the flat output's axis 2 and the 10 dimensions of the second input's axis 2 are mapped to the last 10 dimensions of the flat output's axis 2.
        This is only possible when the remaining axes (here, the leading batch_size axis and axis 1) have the same dimensionalities for all inputs and y.
        Finally, y is reshaped to its final output shape [batch_size] + [2,10,3] and returned.
        
        :param x: A dictionary of [int, tensor] where a key is an index of a connection and a value is a tensor to be processed. This input is assumed to be non-empty and an exception will be raised if the assumption is violated.
        :type x: Dict[int, torch.Tensor]
        """

        # Input validity
        if not isinstance(x, Dict) or not all([isinstance(key, int) and isinstance(value, torch.Tensor) for key, value in x.items()]): 
            raise TypeError(f"The input x to a Merger's forward method should be of type Dict[int, torch.Tensor], not {type(x)}.")
        
        if len(x) == 0: raise ValueError(f"The input x to a Merger's forward method should not be empty.")

        # Initialize output
        batch_size = list(x.values())[0].shape[0]
        dtype = list(x.values())[0].dtype
        device = list(x.values())[0].device
        global y
        y = torch.zeros(size = [batch_size] + self._final_output_shape, dtype=dtype, device=device) # Shape == [batch_size] + final output_shape
        y = torch.flatten(input=y, start_dim=self._output_flatten_axes[0], end_dim=self._output_flatten_axes[1]) # Shape == [batch_size] + flattened output shape

        # Iterate inputs
        y_axis = self._output_flatten_axes[0] # The axis where the new input dimensions will be inserted
        global current_input_tensor
        for connection_index, current_input_tensor in x.items():

            # Flatten x along specified axes
            current_input_tensor = torch.flatten(input=current_input_tensor, 
                                           start_dim=self._connection_index_to_input_flatten_axes[connection_index][0], 
                                           end_dim=self._connection_index_to_input_flatten_axes[connection_index][1])
            
            # Copy them to y
            access_string = "[" + ",".join([":"]*(y_axis)) + ","  + str(self._connection_index_to_output_indices[connection_index]) + "]"
            exec(f"global y, current_input_tensor; y{access_string} = current_input_tensor")

        # Unflatten y
        y = torch.unflatten(input=y, dim=y_axis, sizes=self._final_output_shape[(self._output_flatten_axes[0]-1):(self._output_flatten_axes[1])])

        # Output
        return y 

class ConnectionTransformation(torch.nn.Module):
    """Superclass for a transformation to be placed on a :py:class:`~briann.network.core.Connection`. This default implementation only perform the identity transformation on its input.
    
    :param input_shape: Sets the :py:meth:`~briann.network.core.ConnectionTransformation.input_shape` property.
    :type input_shape: List[int]
    :param output_shape: Sets the :py:meth:`~briann.network.core.ConnectionTransformation.output_shape` property.
    :type output_shape: List[int]
    :return: An instance of this class.
    :rtype: ConnectionTransformation.
    """

    def __init__(self, input_shape: List[int], output_shape: List[int]):

        # Call super
        super().__init__()

        # Properties
        self.input_shape = input_shape
        self.output_shape = output_shape

    @property
    def input_shape(self, ) -> List[int]:
        """:return: The shape of the input to this transformation, disregarding the initial batch axis.
        :rtype: List[int]
        """
        return self._input_shape
    
    @input_shape.setter
    def input_shape(self, new_value: List[int]) -> None:
        
        # Check input validity
        if not isinstance(new_value, List) or not all([isinstance(entry, int) for entry in new_value]): 
            raise TypeError(f"The input_shape of ConnectionTransformation is expected to be a list of ints but was {new_value}.")

        # Update 
        self._input_shape = new_value

    @property
    def output_shape(self, ) -> List[int]:
        """:return: The shape of the output of this transformation, disregarding the initial batch axis.
        :rtype: List[int]
        """
        return self._output_shape
    
    @output_shape.setter
    def output_shape(self, new_value: List[int]) -> None:
        
        # Check output validity
        if not isinstance(new_value, List) or not all([isinstance(entry, int) for entry in new_value]): 
            raise TypeError(f"The output_shape of ConnectionTransformation is expected to be a list of ints but was {new_value}.")

        # Update 
        self._output_shape = new_value

    def forward(self, x):
        return x

class LinearConnectionTransformation(ConnectionTransformation):
    """A simple linear transformation to be placed on a :py:class:`~briann.network.core.Connection`. The input is first flattened along all axes except the first (batch axis) and 
    then passed through a regular :py:class:`torch.nn.Linear` layer before being reshaped to fit the output shape. Construction of this object allow for keyword arguments
    that further configure the linear transformation taken from torch, i.e. bias, device, dtype.
    
    :param input_shape: Sets the :py:meth:`~briann.network.core.ConnectionTransformation.input_shape` property.
    :type input_shape: List[int]
    :param output_shape: Sets the :py:meth:`~briann.network.core.ConnectionTransformation.output_shape` property.
    :type output_shape: List[int]
    """

    def __init__(self, input_shape: List[int], output_shape: List[int], **kwargs) -> None:

        # Super
        super().__init__(input_shape=input_shape, output_shape=output_shape)

        # Choose parameters
        input_dimensionality = torch.prod(input_shape)
        output_dimensionality = torch.prod(output_shape)
        self._linear = torch.nn.Linear(in_features = input_dimensionality, out_features = output_dimensionality, **kwargs)

        # Add callbacks
        bpuc.CallbackManager.add_callback_to_attribute(target_class=LinearConnectionTransformation, target_instance=self, attribute_name='input_shape', callback=LinearConnectionTransformation._on_update_shape)
        bpuc.CallbackManager.add_callback_to_attribute(target_class=LinearConnectionTransformation, target_instance=self, attribute_name='output_shape', callback=LinearConnectionTransformation._on_update_shape)

    def _on_update_shape(obj: ConnectionTransformation, name: str, value: List[int]) -> None:
        """This method is a callback that adjusts the model parameters of the transformation of self whenever the :py:meth:`~briann.network.core.ConnectionTransformation.input_shape` or :py:meth:`~briann.network.core.ConnectionTransformation.output_shape` is updated.
        
        :param obj: The object on which the input_shape is updated.
        :type obj: :py:class:`~briann.network.core.ConnectionTransformation`
        :param name: The name of the attribute (i.e. 'input_shape' or 'output_shape) to be updated.
        :type name: str
        :param value: The new shape of the input or output, disregarding the batch_size as axis 0.
        :type value: List[int]
        :rtype: None"""

        # Input validity
        if name not in ["input_shape", "output_shape"]: raise ValueError(f"The callback LinearConnectionTransformation should only be added to the attributes 'input_shape' and 'output_shape', not to {name}.")
        
        # Extract relevant information
        if name == 'input_shape':
            input_dimensionality = torch.prod(value)
            output_dimensionality = obj._linear.out_features
        elif name == 'output_shape':
            input_dimensionality = obj._linear.in_features
            output_dimensionality = torch.prod(value)
        
        # Update parameters of self
        obj._linear = torch.nn.Linear(in_features=input_dimensionality, 
                                        out_features=output_dimensionality, 
                                        bias=obj._linear.bias != None, 
                                        device=obj._linear.device, 
                                        dtype=obj._linear.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validity
        if not len(x.shape) > 1 and x.shape[1:] == self.input_shape: raise ValueError(f"The input to connection transformation {self} was expected to be of shape {self.input_shape} (disregarding the initial batch-axis), but was found to {x.shape[1:]}.")
        
        # Transform
        x = torch.flatten(input=x, start_dim=1)
        x = self._linear(x)
        y = torch.unflatten(input=x, dim=1, sizes=self.output_shape)

        # Output
        return y

class Connection(torch.nn.Module):
    """A connection between two :py:class:`~briann.network.coreArea` objects. This is analogous to a neural tract between areas of a biological neural network that 
    not only sends information but also converts it between the reference frames of the input and output area. It thus has a 
    :py:meth:`~briann.network.core.Connection.transformation` that is applied to the input before it is sent to the target area. For biological plausibility, 
    the transformation should be a simple linear transformation, for instance a :py:class:`torch.nn.Linear` layer.
    
    :param index: Sets the :py:attr:`~briann.network.core.Area.index` of this area.
    :type index: int
    :param from_area_index: Sets the :py:meth:`~briann.network.core.Connection.from_area_index` of this connection. 
    :type from_area_index: int
    :param to_area_index: Sets the :py:meth:`~briann.network.core.Connection.to_area_index` of this connection. 
    :type to_area_index: int
    :param input_time_frame_accumulator: Used to set :py:meth:`~briann.network.core.Connection.input_time_frame_accumulator` of self.
    :type input_time_frame_accumulator: :py:class:`~briann.network.core.TimeFrameAccumulator`
    :param transformation: Sets the :py:meth:`~briann.network.core.Connection.transformation` of the connection.
    :type transformation: :py:class:`~briann.network.core.ConnectionTransformation`
    """

    def __init__(self, 
                 index: int, 
                 from_area_index: int, 
                 to_area_index: int, 
                 input_time_frame_accumulator: TimeFrameAccumulator, 
                 transformation: ConnectionTransformation) -> None:
        
        # Call the parent constructor
        super().__init__()

        # Set Properties        
        self.index = index # Must be set first
        self.from_area_index = from_area_index
        self.to_area_index = to_area_index
        self.input_time_frame_accumulator = input_time_frame_accumulator
        self.transformation = transformation
        
    @property
    def index(self) -> int:
        """:return: The index used to identify this connection in the overall model.
        :rtype: int"""
        return self._index
    
    @index.setter
    def index(self, new_value: int) -> None:

        # Check input validity
        if not (isinstance(new_value, int)):
            raise TypeError(f"The index of Connection {self.index} must be an int but was {type(new_value)}.")
        
        # Set property
        self._index = new_value

    @property
    def from_area_index(self) -> int:
        """:return: The index of the area that is the source of this connection.
        :rtype: int
        """
        return self._from_area_index
    
    @from_area_index.setter
    def from_area_index(self, new_value: int) -> None:
        
        # Check input validity
        if not isinstance(new_value, int):
            raise TypeError(f"The from_area_index of Connection {self.index} must be an int but was {type(new_value)}.")
        
        # Set property
        self._from_area_index = new_value

    @property
    def to_area_index(self) -> int:
        """:return: The index of the area that is the target of this connection.
        :rtype: int
        """
        return self._to_area_index
    
    @to_area_index.setter
    def to_area_index(self, new_value: int) -> None:
        
        # Check input validity
        if not isinstance(new_value, int):
            raise TypeError(f"The to_area_index of connection {self.index} must be an int but was {type(new_value)}.")
        
        # Set property
        self._to_area_index = new_value

    @property
    def input_time_frame_accumulator(self) -> TimeFrameAccumulator:
        """:return: The time frame accumulator that stores the input of the connection.
        :rtype: :py:class:`~briann.network.core.TimeFrameAccumulator`
        """
        return self._input_time_frame_accumulator

    @input_time_frame_accumulator.setter
    def input_time_frame_accumulator(self, new_value: TimeFrameAccumulator) -> None:
        
        # Check input validity
        if not isinstance(new_value, TimeFrameAccumulator):
            raise TypeError(f"The input_time_frame_accumulator of Connection {self.index} must be a TimeFrameAccumulator but was {type(new_value)}.")
        
        # Set property
        self._input_time_frame_accumulator = new_value
    
    def forward(self, current_time: float) -> TimeFrame:
        """Reads the current state of the :py:meth:`~briann.network.core.Connection.time_frame_accumulator` and applies the :py:meth:`~briann.network.core.Connection.transformation` to it. 

        :param current_time: The current time in the simulation.
        :type current_time: float  
        :return: The produced time frame.
        :rtype: :py:class:`~briann.network.core.TimeFrame`
        """

        # Read input
        input_state = self.input_time_frame_accumulator.time_frame(current_time=current_time).state
        try:self.transformation(input_state)
        except Exception as e:
            bla=1
        # Apply the transformation to the time frame
        transformed_state = self.transformation(input_state)
        
        # Create a new time frame with the transformed state
        new_time_frame = TimeFrame(state=transformed_state, time_point=current_time)
    
        # Output
        return new_time_frame

    def __repr__(self) -> str:
        """Returns a string representation of the connection."""
        return f"Connection(index={self._index}), from_area_index={self._from_area_index}, to_area_index={self._to_area_index})"
  
class Area(torch.nn.Module):
    """An area corresponds to a small population of neurons that jointly hold a representation in the area's :py:meth:`~briann.network.core.Area.output_time_frame_accumulator`.
    Given a time-point t and a set S of areas that should be updated at t, the caller should update the areas' states in two consecutive loops over S. The first loop
    should call the :py:meth:`~briann.network.core.Area.collect_inputs` method on each area to make it collect, sum and buffer its inputs from the overall network. Then, in the second loop, the 
    :py:meth:`~briann.network.core.Area.forward` method should be called on each area of S to sum the buffered inputs and apply the area's :py:meth:`~briann.network.core.Area.transformation`. 
    This splitting of input collection and forward transformation allows for parallelization of areas.
    
    :param index: Sets the :py:attr:`~briann.network.core.Area.index` of this area.
    :type index: int
    :raises ValueError: If the index is not a non-negative integer.
    :param output_time_frame_accumulator: Sets the :py:meth:`~briann.network.core.Area.output_time_frame_accumulator` of self.
    :type output_time_frame_accumulator: :py:class:`~briann.network.core.TimeFrameAccumulator`
    :param input_connections: Sets the :py:meth:`~briann.network.core.Area.input_connections` of this area.
    :type input_connections: List[:py:class:`~briann.network.core.Connection`]
    :param input_shape: Sets the :py:meth:`~briann.network.core.Area.input_shape` of this area.
    :type input_shape: List[int]
    :param output_shape: Sets the :py:meth:`~briann.network.core.Area.output_shape` of this area.
    :type output_shape: List[int]
    :param output_connections: Sets the :py:meth:`~briann.network.core.Area.output_connections` of this area.
    :type output_connections: List[:py:class:`~briann.network.core.Connection`]
    :param merger: Sets the :py:meth:`~briann.network.core.Area.merger` property of self.
    :type merger: :py:class:`~briann.network.core.Merger`
    :param transformation: Sets the :py:meth:`~briann.network.core.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~briann.network.core.Area.update_rate` of this area.
    :type update_rate: float
    """

    def __init__(self, index: int, 
                 output_time_frame_accumulator: TimeFrameAccumulator, 
                 input_connections: List[Connection], 
                 input_shape: List[int],
                 output_shape: List[int],
                 output_connections: List[Connection],
                 merger: Merger,
                 transformation: torch.nn.Module,
                 update_rate: float) -> None:
        
        # Call the parent constructor
        super().__init__()
        
        # Ensure input validity
        if not isinstance(input_shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise TypeError(f"The input_shape of area {index} must be a list of positive integers but was {input_shape}.")
        
        if not isinstance(output_shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in output_shape):
            raise TypeError(f"The output_shape of area {index} must be a list of positive integers but was {output_shape}.")
        if not output_shape == list(output_time_frame_accumulator._time_frame.state.shape[1:]):
            raise ValueError(f"The output_shape of area {index} must match the shape of the state of its output_time_frame_accumulator but was {output_shape} and {output_time_frame_accumulator._time_frame.state.shape[1:].as_list()}, respectively.")
        
        # Set properties
        self.index = index # Must be set first
        self.output_time_frame_accumulator = output_time_frame_accumulator
        self.input_connections = input_connections
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.output_connections = output_connections
        self.merger = merger
        
        # Check input validity
        if not isinstance(transformation, torch.nn.Module):
            raise TypeError(f"The transformation of area {self.index} must be a torch.nn.Module object.")
        
        self._transformation = transformation # With torch, it is not possible to use the regular property setter/ getter, hence, the transformation is set once here manually and then kept private
        
        self.update_rate = update_rate
        self._update_count = 0
        self._input_state = None # Will store the buffered input states updated by collect_inputs

    @property
    def index(self) -> int:
        """:return: The index used to identify this area in the overall model.
        :rtype: int"""
        return self._index
    
    @index.setter
    def index(self, new_value: int) -> None:
        
        # Check input validity
        if not isinstance(new_value, int):
            raise TypeError("The index must be an int.")
        if not new_value >= 0:
            raise ValueError(f"The index must be non-negative but was set to {new_value}.")
        
        # Set property
        self._index = new_value

        # Adjust input connections
        if hasattr(self, "_input_connections"):
            for connection in self.input_connections:
                connection.to_area_index = new_value

        # Adjust output connections
        if hasattr(self, "_output_connections"):
            for connection in self.output_connections:
                connection.from_area_index = new_value

    @property
    def output_time_frame_accumulator(self) -> TimeFrameAccumulator:
        """:return: The time-frame accumulator of this area. This holds the output state of the area which will be made available to other areas via :py:class:`~briann.network.core.Connection`.
        :rtype: :py:class:`~briann.network.core.TimeFrameAccumulator`"""
        return self._output_time_frame_accumulator

    @output_time_frame_accumulator.setter
    def output_time_frame_accumulator(self, new_value: TimeFrameAccumulator) -> None:
        
        # Check input validity
        if not isinstance(new_value, TimeFrameAccumulator):
            raise TypeError(f"The time_frame_accumulator must be a TimeFrameAccumulator.")
        
        # Set property
        self._output_time_frame_accumulator = new_value

        # Update output connections
        if hasattr(self, "_output_connections"):
            for connection in self.output_connections:
                connection.input_time_frame_accumulator = new_value

    @property
    def input_connections(self) -> Set[Connection]:
        """:return: A set of :py:class:`~briann.network.core.Connection` objects projecting to this area.
        :rtype: Set[Connection]
        """
        return self._input_connections

    @input_connections.setter
    def input_connections(self, new_value: Set[Connection]) -> None:
        # Check input validity
        if not isinstance(new_value, Set):
            raise TypeError(f"The input_connections for area {self.index} must be a set of :py:class:`~briann.network.core.Connection` objects projecting to area {self.index}.")
        if not all(isinstance(connection, Connection) for connection in new_value):
            raise TypeError(f"All values in the input_connections set of area {self.index} must be Connection objects projecting to area {self.index}.")
        
        # Set property
        self._input_connections = new_value 

    @property
    def input_shape(self) -> int:
        """:return: The shape of the input to this area for a single instance (i.e. excluding the batch-dimension that is assumed to be at index 0 of the actual input).
        :rtype: int
        """
        return self._input_shape

    @property
    def output_shape(self) -> int:
        """:return: The shape of the output of this area for a single instance (i.e. excluding the batch-dimension that is assumed to be at index 0 of the actual output). The output is the state held in the :py:meth:`~briann.network.core.Area.output_time_frame_accumulator` and hence has same shape.
        :rtype: int
        """
        return self._output_shape

    @property
    def output_connections(self) -> Set[Connection]:
        """:return: A set of :py:class:`~briann.network.core.Connection` objects projecting from this area. 
        :rtype: Set[Connection]
        """
        return self._output_connections

    @output_connections.setter
    def output_connections(self, new_value: Set[Connection]) -> None:

        # Check input validity
        if not isinstance(new_value, Set):
            raise TypeError(f"The output_connections for area {self.index} must be a set of :py:class:`~briann.network.core.Connection` objects projecting from area {self.index}.")
        if not all(isinstance(connection, Connection) for connection in new_value):
            raise TypeError(f"All values in the output_connections set of area {self.index} must be Connection objects projecting from area {self.index}.")
        if 0 < len(new_value):
            time_frame_accumulator = list(new_value)[0].input_time_frame_accumulator
            for connection in list(new_value)[1:]:
                if not connection.input_time_frame_accumulator == time_frame_accumulator:
                    raise ValueError("When setting the output_connections of an area, they must all have the same input_time_frame_accumulator")  

        # Set property
        self._output_connections = new_value

        # Set output_time_frame_accumulator
        if 0 < len(new_value):
            self._output_time_frame_accumulator = list(new_value)[0].input_time_frame_accumulator
    
    @property
    def merger(self) -> Merger:
        """:return: This merger is used to merge the input signals in the :py:meth:`~briann.network.core.Area.collect_inputs` method.
        :rtype: :py:class:`~briann.network.core.Merger`"""

        return self._merger
    
    @merger.setter
    def merger(self, new_value) -> None:

        # Input validity
        if new_value != None and not isinstance(new_value, Merger): raise TypeError(f"When setting the merger of area {self.index}, an object of type Merger should be given.")

        # Set property
        self._merger = new_value

    @property
    def update_rate(self) -> float:
        """:return: The update-rate of this area.
        :rtype: float"""
        return self._update_rate
    
    @update_rate.setter
    def update_rate(self, new_value: float) -> None:

        # Check input validity
        if not isinstance(new_value, float) and not isinstance(new_value, int):
            raise TypeError(f"The update_rate of area {self.index} has to be a float.")
        if not new_value > 0:
            raise ValueError(f"The update_rate of area {self.index} has to be positive.")
        
        # Set property
        self._update_rate = (float)(new_value)

    @property
    def update_count(self) -> int:
        """:return: Counts how many times this area was updated during the simulation.
        :rtype: int"""
        return self._update_count
    
    def collect_inputs(self, current_time: float) -> None:
        """Calls the :py:meth:`~briann.network.core.Connection.forward` method of all incoming connections to get the current inputs, sums them up and buffers 
        the result for later use by the :py:meth:`~briann.network.core.Area.forward` method. Since the inputs are summed, it is necessary that they are all of the same shape. 
        
        :param current_time: The current time of the simulation used to time-discount the states of the input areas.
        :type current_time: float
        :rtype: None
        """

        # Initalize input state
        self._input_state = None
        
        # Collect inputs
        x = dict({(connection.index, connection.forward(current_time=current_time).state) for connection in self.input_connections})
        
        # Accumulate inputs
        self._input_state = self.merger.forward(x=x)

    def forward(self) -> None:
        """Assuming :py:meth:`~briann.network.core.Area.collect_inputs` has been run on all areas of the simulation just beforehand, this method passes the buffered inputs through
        the `:py:meth:`~briann.network.core.Area.transformation` of self (if exists) and passes the result to the :py:meth:`~briann.network.core.TimeFrameAccumulator.accumulate` of self.
        """

        # Determine current time
        self._update_count += 1
        current_time = self._update_count / self.update_rate

        # Retrieve inputs
        if self._input_state == None:
            raise ValueError(f"The input_states of area {self.index} are None. Run collect_inputs() on all areas before calling forward().")
        new_state = self._input_state
        self._input_state = None

        # Apply transformation to the states
        if not self._transformation == None: new_state = self._transformation.forward(new_state)

        # Create and accumulate a new time-frame for the current state
        new_time_frame = TimeFrame(state=new_state, time_point=current_time)
        self._output_time_frame_accumulator.accumulate(time_frame=new_time_frame)        

        # Notify subscribers
        if hasattr(self, "_subscribers"):
            new_time_frame = self.output_time_frame_accumulator.time_frame(current_time=current_time)
            for subscriber in self._subscribers:
                subscriber.on_state_update(area_index=self.index, time_frame=new_time_frame)

    def reset(self) -> None:
        """Resets the area to its initial state. This should be done everytime a new trial is simulated."""
        
        # Reset the time-frame accumulator
        self._output_time_frame_accumulator.reset()
        
        # Reset the update count
        self._update_count = 0

        # Notify subscribers
        if hasattr(self, "_subscribers"):
            new_time_frame = self._output_time_frame_accumulator.time_frame(current_time=0.0)
            for subscriber in self._subscribers:
                subscriber.on_state_update(area_index=self.index, time_frame=new_time_frame)

    def __repr__(self) -> str:
        """Returns a string representation of the area."""
        return f"Area(index={self._index}, update_rate={self._update_rate}, update_count={self._update_count})"

class Source(Area):
    """The source :py:class:`~briann.network.core.Area` is a special area because it streams the input to the other areas. In order to set it up for the simulation of a trial,
    load stimuli via the :py:meth:`~briann.network.core.Source.load_stimulus_batch method. Then, during each call to the :py:meth:`~briann.network.core.Area.collect_inputs` method, one :py:class:`~briann.network.core.TimeFrame` 
    will be taken from the stimuli and held in a bffer. Upon calling the :py:meth:`~briann.network.core.Area.forward` method, that time-frame will be placed in the
    :py:meth:`~briann.network.core.Area.TimeFrameAccumulator`, so that it can be read by other areas. Once the time frames are all streamed, the source area will no longer add new
    time-frames to the accumulator and hence its representation will simply decay over time.

    :param index: Sets the :py:attr:`~briann.network.core.Area.index` of this area.
    :type index: int
    :param output_time_frame_accumulator: Sets the :py:meth:`~briann.network.core.Area.time_frame_accumulator` of this area. 
    :type output_time_frame_accumulator: :py:class:`~briann.network.core.TimeFrameAccumulator`
    :param output_shape: Sets the :py:meth:`~briann.network.core.Area.output_shape` of this area.
    :type output_shape: List[int]
    :param output_connections: Sets the :py:meth:`~.Areabriann.network.core.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`~briann.network.core.Connection`]
    :param update_rate: Sets the :py:meth:`~briann.network.core.Area.update_rate` of this area.
    :type update_rate: float
    """

    def __init__(self, index: int, output_time_frame_accumulator: TimeFrameAccumulator, output_shape: List[int], output_connections: Dict[int, Connection], update_rate: float) -> None:

        # Call the parent constructor
        super().__init__(index=index,
                         output_time_frame_accumulator=output_time_frame_accumulator,
                         input_connections=set([]),
                         input_shape=[],
                         output_shape=output_shape,
                         output_connections=output_connections,
                         merger=None,
                         transformation=torch.nn.Identity(),
                         update_rate=update_rate)
        
        # Set properties
        self._stimulus_batch = None

    @property
    def stimulus_batch(self) -> Deque[TimeFrame]:
        """The stimuli that are currently loaded in the source area. This is a deque of :py:class:`~briann.network.core.TimeFrame` objects that are to be processed by the model.
        
        :return: The stimuli.
        :rtype: Deque[:py:class:`~briann.network.core.TimeFrame`]
        """
        return self._stimulus_batch

    def load_next_stimulus_batch(self, X: torch.Tensor) -> None:
        """This method loads the next batch of stimuli that will be streamed to the other model areas during the simulation. 

        :param X: A tensor of shape [batch_size, time_frame_count, ...] where the first axis corresponds to instances in the batch and the second axis to time-frames.
        :type X: :py:class:`torch.Tensor`
        :raises Exception: if self.data_loader is None.
        :raises StopIteration: if the data_loader is empty.
        """
        
        # Check input validity
        if not isinstance(X, torch.Tensor): raise TypeError(f"Input X was expected to be a torch.Tensor, but is {type(X)}.")
        if not len(X.shape) >= 2: raise ValueError(f"Input X was expected to have at least 2 axes, namely the first for instances of a batch and the second for time-frames, but it has {len(X.shape)} axes.")
        if len(X.shape) == 2: X = X[:,:,torch.newaxis]

        # Convert to batch of stimulus time-frames
        self._stimulus_batch = Deque([])
        time_frame_count = X.shape[1]
        for t in range(time_frame_count):
            time_frame = TimeFrame(state=X[:,t,:], time_point = (t)/self.update_rate)
            self._stimulus_batch.appendleft(time_frame)

        # Load first time-frame
        self._update_count -= 1 # This will be incremented again in the forward method and then the first data point corresponds to the default update count
        self.collect_inputs(current_time=0.0)
        self.forward()

    def collect_inputs(self, current_time: float) -> None:
        """Pops the next :py:class:`~briann.network.core.TimeFrame` from :py:meth:`~briann.network.core.Source.stimulus_batch` or generates an array of zeros if the stimulus stream is over. Either way, the result is buffered internally to be made available to other areas upon calling :py:meth:`~briann.network.core.Area.forward`.
        
        :param current_time: The current time of the simulation.
        :type current_time: float
        :raises ValueError: if the `current_time` is not equal to the time of the popped :py:class:`~briann.network.core.TimeFrame`.
        :rtype: None
        """

        # Get the next time frame
        if len (self._stimulus_batch) > 0:
            new_time_frame = self._stimulus_batch.pop()

            # Ensure input validity
            if not current_time == new_time_frame.time_point: raise ValueError(f"The collect_inputs method of Source {self.index} expected to be called next at time-point {new_time_frame.time_point} but was called at time-point {current_time}.")

            # Store a reference to the current time-frame for later
            self._input_state = new_time_frame.state
            
        else:
            # No more time-frames to pop, simply create array of zeros to be added to the output time-frame accumulator in forward()
            current_time_frame = self.output_time_frame_accumulator.time_frame(current_time=current_time)
            self._input_state = torch.zeros_like(current_time_frame.state)

class Target(Area):
    """This class is a subclass of :py:class:`~briann.network.core.Area` and has the same functionality as a regular area except that it has no output connections.

    :param index: Sets the :py:attr:`~briann.network.core.Area.index` of this area.
    :type index: int
    :param output_time_frame_accumulator: Sets the :py:meth:`~briann.network.core.Area.output_time_frame_accumulator` of self.
    :type output_time_frame_accumulator: :py:class:`~briann.network.core.TimeFrameAccumulator`
    :param input_connections: Sets the :py:meth:`~briann.network.core.Area.input_connections` of this area.
    :type input_connections: List[:py:class:`~briann.network.core.Connection`]
    :param input_shape: Sets the :py:meth:`~briann.network.core.Area.input_shape` of this area.
    :type input_shape: List[int]
    :param output_shape: Sets the :py:meth:`~briann.network.core.Area.output_shape` of this area.
    :type output_shape: List[int]
    :param merger: Sets the :py:meth:`~briann.network.core.Area.merger` property of self.
    :type merger: :py:class:`~briann.network.core.Merger`
    :param transformation: Sets the :py:meth:`~briann.network.core.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~briann.network.core.Area.update_rate` of this area.
    :type update_rate: float
    """

    def __init__(self, index: int, 
                 output_time_frame_accumulator: TimeFrameAccumulator, 
                 input_connections: List[Connection],
                 input_shape: List[int],
                 output_shape: List[int],
                 merger: Merger,
                 transformation: torch.nn.Module, 
                 update_rate: float) -> None:
        
        # Cqll to super
        super().__init__(index=index, 
                 output_time_frame_accumulator=output_time_frame_accumulator, 
                 input_connections=input_connections, 
                 input_shape=input_shape,
                 output_shape = output_shape,
                 output_connections=None,
                 merger=merger,
                 transformation=transformation, 
                 update_rate=update_rate)

        # Set properties
        self._output_states = deque([])
    

    @Area.output_connections.setter
    def output_connections(self, new_value: List[Connection]) -> None:
        if new_value != None:
            raise ValueError("A Target area does not accept any output connections.")

    def forward(self) -> None:
        """Assuming :py:meth:`~briann.network.core.Area.collect_inputs` has been run on all areas of the simulation just beforehand, this method passes the buffered inputs through
        the `:py:meth:`~briann.network.core.Area.transformation` of self (if exists) and passes the result to the :py:meth:`~briann.network.core.TimeFrameAccumulator.accumulate` of self.
        """

        # Call to super
        super().forward()

        # Collect state time-frame
        current_time = self._update_count / self.update_rate
        new_time_frame = self.output_time_frame_accumulator.time_frame(current_time=current_time)
        self._output_states.append(new_time_frame.state)

    def extract_current_output_batch(self, final_states_only: bool = True) -> torch.Tensor:
        """Extracts the current output batch held by this target area. This is done by collecting all time-frames obtained by :py:meth:`~briann.network.core.Target.forward` and held in buffer and stacking them into a tensor
        that is returned. After extraction, the internal buffer of time-frames is cleared.
        
        :param final_states_only: Indicates whether only the final, i.e. current output states shall be returned or the entire sequence.
        :type final_states_only: bool, optional, defaults to True
        :return: The output batch held by this target area. This is a tensor of shape [batch_size, time_frame_count, ...], where ... is the shape of the state of a single instance's time-frame of this area
        :rtype: :py:class:`torch.Tensor`
        """

        # Ensure state validity
        if not self._update_count == len(self._output_states): raise ValueError(f"While extracting the output time-frame, the internal update count of Target area {self.index} was found to not match the number of held output time-frames. Reset the area and try again.")
        
        # Early exit
        if final_states_only:
            self._output_states.clear()
            return self.output_time_frame_accumulator.time_frame(current_time=self.update_count * self.update_rate)
        
        # Collect time-frames
        time_frame_count = self.update_count
        if time_frame_count == 0:
            raise ValueError(f"The Target area {self.index} does not hold any output time-frames to extract.")
        
        # Stack into tensor
        output_states = [self._output_states.popleft().state[:, torch.newaxis, :] for _ in range(time_frame_count)] # Popping entries clears the deque
        output_batch = torch.stack(output_states, dim=1)

        # Output
        return output_batch
    

class BrIANN(torch.nn.Module):
    """This class functions as the network that holds together all its :py:class:`~briann.network.core.Area`'s and :py:class:`~briann.network.core.Connection`'s. Its name abbreviates Brain Inspired Artificial Neural Networks. 
    To use it, one should provide a configuration dictionary from which all components can be loaded. 
    Then, for each batch, one should call :py:meth:`~briann.network.core.BrIANN.load_next_stimulus_batch`.
    Once a batch is loaded, the processing can be simulated for as long as the caller intends (ideally at least for as long as the
    :py:class:`~briann.network.core.Source` areas provide :py:class:`~briann.network.core.TimeFrame`'s) using the :py:meth:`~briann.network.core.BrIANN.step` method.
    In order to get a simplified networkx representation which contains information about the large-scale network topology (:py:class:`~briann.network.core.Area`'s and :py:class:`~briann.network.core.Connection`'s),
    one can use :py:meth:`~briann.network.core.BrIANN.get_topology`.
    
    :param configuration: A configuration in the form of a dictionary.
    :type configuration: Dict[str, Any]
    """

    def __init__(self, name, areas: List[Area], connections: List[Connection]) -> None:
        
        # Call the parent constructor
        super().__init__()
        
        # Set properties
        self.name = name
        self._areas = torch.nn.ModuleList(areas)
        self._connections = torch.nn.ModuleList(connections)
        self._current_simulation_time = 0.0
        """:return: The time that has passed since the start of the simulation. It is updated after each step of the simulation.
        :rtype: float
        """

    @property
    def areas(self) -> torch.nn.ModuleList:
        """:return: The set of areas held by self.
        :rtype: torch.nn.ModuleList
        """

        return self._areas

    def get_area_indices(self) -> Set[int]:
        """:return: The set of indices of the areas stored internally.
        :rtype: Set[int]
        """
        return set([area.index for area in self._areas])
    
    def get_area_at_index(self, index: int) -> Area:
        """:return: The area with given `index`.
        :rtype: :py:class:`~briann.network.core.Area`
        :raises ValueError: If self does not store an area of given `index`
        """
        
        # Check input validity
        if not isinstance(index, int): raise TypeError(f"The area index was expected to be of type int but was {type(index)}.")
        if not index in self.get_area_indices(): raise ValueError(f"This BrIANN object does not hold an area with index {index}.")
        
        # Collect
        result = None
        for area in self._areas:
            if area.index == index: result = area

        # Output
        return result
    
    @property
    def connections(self) -> torch.nn.ModuleList:
        """:return: The set of internally stored :py:class:`~briann.network.core.Connection`.
        :rtype: torch.nn.ModuleList
        """
        return self._connections
    
    def get_connections_from(self, area_index: int) -> Set[Connection]:
        """:return: A set of :py:class:`~briann.network.core.Connection` objects that are the output connections of the area with the given index. 
        :rtype: Set[:py:class:`~briann.network.core.Connection`]
        """

        # Compile
        result = [None] * len(self._connections)
        i = 0
        for connection in self._connections:
            if connection.from_area_index == area_index: 
                result[i] = connection
                i += 1

        # Output
        return set(result[:i])
        
    def get_connections_to(self, area_index: int) -> Set[Connection]:
        """:return: A set of :py:class:`~briann.network.core.Connection` objects that are the input connections to the area with the given index. 
        :rtype: Set[:py:class:`~briann.network.core.Connection`]
        """

        # Compile
        result = [None] * len(self._connections)
        i = 0
        for connection in self._connections:
            if connection.to_area_index == area_index: 
                result[i] = connection
                i += 1

        # Output
        return set(result[:i])
    
    @property
    def current_simulation_time(self) -> float:
        """:return: The time that has passed since the start of the simulation. It is updated after each step of the simulation.
        :rtype: float
        """
        return self._current_simulation_time

    def get_topology(self) -> nx.DiGraph:
        """Converts the BrIANN network to a NetworkX DiGraph where each node is simply the :py:meth:`~briann.network.core.Area.index` of a corresponding :py:class:`~briann.network.core.Area`
        and each edge is simply the triplet (*u*,*v*) where *u* is the :py:meth:`~briann.network.core.Connection.from_index`, *v* the :py:meth:`~briann.network.core.Connection.to_index` of the corresponding :py:class:`~briann.network.core.Connection`.
        
        :return: A NetworkX DiGraph representing the BrIANN network.
        :rtype: nx.DiGraph
        """

        # Create a directed graph
        G = nx.DiGraph()
         
        # Add nodes for each area
        area_indices = sorted(self.get_area_indices())
        for area_index in area_indices:
            area = self.get_area_at_index(index=area_index)
            G.add_node(area, index=area_index)
        
        # Add edges for each connection
        for connection in self.connections:
            from_area = self.get_area_at_index(index=connection.from_area_index)
            to_area = self.get_area_at_index(index=connection.to_area_index)
            G.add_edge(u_of_edge=from_area, v_of_edge=to_area)
        
        # Output
        return G

    def load_next_stimulus_batch(self, X: torch.Tensor | Dict[int, torch.Tensor]) -> None:
        """This method resets the :py:meth:`~briann.network.core.BrIANN.current_simulation_time` and all areas. It also makes the :py:class:`~briann.network.core.Source` areas
        load their corresponding next batch of stimuli. It thus assumes that all source areas have a valid :py:meth:`~briann.network.core.Source.data_loader` set
        and that the data loaders are in sync with each other and non-empty.

        :param X: A tensor of shape [batch_size, time_frame_count, ...] or a Dict[int, :py:class:`torch.Tensor`] where the tensor's first axis corresponds to instances in the batch and the second axis to time-frames. If a dictionary is provided, then each key is an index of a source area and the value is the corresponding input tensor.
        :type X: :py:class:`torch.Tensor` | Dict[int, :py:class:`torch.Tensor`]
        :rtype: None
        """
        
        # Reset the states of all areas
        for area in self.areas:
            area.reset()

        # Load the next batch of stimuli into the source areas
        for area in self.areas:
            if isinstance(area, Source):
                if isinstance(X, Dict):
                    if area.index not in X.keys():
                        raise ValueError(f"When providing a dictionary of inputs to load_next_stimulus_batch, the dictionary must contain an entry for each source area. However, source area {area.index} is missing.")
                    area.load_next_stimulus_batch(X=X[area.index])
                else:
                    area.load_next_stimulus_batch(X=X)

        # Reset the simulation time
        self._current_simulation_time = 0.0

    def step(self) -> Set[Area]:
        """Performs one step of the simulation by finding the set of areas due to be updated next and calling their :py:meth:`~briann.network.core.Area.collect_inputs` and
        :py:meth:`~briann.network.core.Area.forward` method to make them process their inputs. 
        This method needs to be called repeatedly to step through the simulation. The simulation does not have an internally checked stopping condition,
        meaning this step method can be called indefinitely, even if the sources already ran out of stimuli. 
        The caller of this method thus needs to determine when to stop the simulation.

        :return: The set of areas that were updated within this step.
        :rtype: Set[:py:class:`~briann.network.core.Area`]
        """
        
        # Find the areas that are due next
        due_areas = set([])
        min_time = sys.float_info.max
        for area in self._areas:
            area_next_time = (area.update_count +1) / area.update_rate # Add 1 to get the time of the area's next frame 
            if area_next_time == min_time: # Current area belongs to current set of due areas
                due_areas.add(area)
            elif area_next_time < min_time: # Current area is due sooner 
                due_areas = set([area])
                min_time = area_next_time
            
        # Update the simulation time
        self._current_simulation_time = min_time

        # Make all areas collect their inputs
        for area in due_areas: area.collect_inputs(current_time=self.current_simulation_time)

        # Make all areas process their inputs
        for area in due_areas: area.forward()

        # Outputs
        return due_areas

    def extract_current_output_batch(self, final_states_only: bool = True) -> torch.Tensor | Dict[int, torch.Tensor]:
        """Extracts the current output batch(es) held by the target area(s) of this model. This is done for each target area by either collecting the target area's current states
        (if `final_states_only` is True) or the entire sequence of output frames (if `final_states_only` is False). 
        After extraction, the target area's internal buffer of time-frames is cleared.
        
        :param final_states_only: Indicates whether only the final, i.e. current output states shall be returned or the entire sequence.
        :type final_states_only: bool, optional, defaults to True
        :return: If there is only a single target area, the output will be a torch.Tensor. Otherwise, it will be a dictionary for which a key is a target area's index and the corresponding value a tensor. Each such tensor is of shape [batch_size, time_frame_count, ...], where ... is the shape of the state of a single instance's time-frame of the corresponding target area.
        :rtype: :py:class:`torch.Tensor` | Dict[int, :py:class:`torch.Tensor`]
        """

        # Input validity
        if not isinstance(final_states_only, bool): raise TypeError(f"The input final_states_only was expected to be of type bool but was {type(final_states_only)}.")

        # Collect outputs
        outputs = dict({})
        for area in self._areas:
            if isinstance(area, Target):
                outputs[area.index] = area.extract_current_output_batch(final_states_only=final_states_only)

        # Return
        if len(outputs.items) == 1:
            return list(outputs.values())[0]
        else:
            return outputs

    def __repr__(self) -> str:
        string = "BrIANN\n"

        for area in self._areas: 
            string += f"{area}\n"
            for connection in self.get_connections_from(area_index=area.index):
                string += f"\t{connection}\n"

        return string


