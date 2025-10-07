"This module collects all necessary components to build a BrIANN model."
import torch
from typing import List, Dict, Deque, Set, Any
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import area_transformations as bpnat
from src.briann.python.training import data_management as bptdm
import networkx as nx
from briann.python.utilities import callbacks as bpuc

class TimeFrame():
    """A time-frame in the simulation that holds a temporary state of an :py:class:`.Area`. 

    :param state: Sets the :py:attr:`~.TimeFrame.state` of this time frame.
    :type state: :py:class:`torch.Tensor`
    :param time_point: Sets the :py:attr:`~.TimeFrame.time_point` of this time frame.
    :type time_point: float
    """
    
    def __init__(self, state: torch.Tensor, time_point: float) -> "TimeFrame":
        
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
        """:return: The time point at which this time frame's :py:meth:`~.TimeFrame.state` occured.
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
    own time-frame using the :py:meth:`~.TimeFrameAccumulator.accumulate` function. An important feature of the accumulator is that during
    every update, the currently stored information decays according to the provided `decay_rate` and the time since the last update. 
    This is done to ensure that older information has less influence on the current state of the accumulator than new information.

    :param initial_time_frame: Sets the :py:attr:`~.TimeFrameAccumulator.initial_time_frame` and :py:attr:`~.TimeFrameAccumulator.time_frame` of this time frame accumulator.
    :type initial_time_frame: :py:class:`.TimeFrame`
    :param decay_rate: Sets the :py:meth:`~.TimeFrameAccumulator.decay_rate` property of self.
    :type decay_rate: float
    :return: A new time-frame accumulator.
    :rtype: :py:class:`.TimeFrameAccumulator`
    """

    def __init__(self, initial_time_frame: TimeFrame, decay_rate: float) -> "TimeFrameAccumulator":
           
        # Set initial time-frame and time-frame
        if not isinstance(initial_time_frame, TimeFrame):
            raise TypeError(f"The initial_time_frame was expected to be a TimeFrame but was {type(initial_time_frame)}.")
        self._time_frame = initial_time_frame
        self._initial_time_frame = initial_time_frame

        # Set decay rate
        self.decay_rate = decay_rate
    
    @property
    def decay_rate(self) -> float:
        """:return: The rate taken from the interval [0,1] at which the energy of the :py:meth:`~.TimeFrame.state` of :py:meth:`~.TimeFrameAccumulator.time_frame` decays as time passes. This rate is recommended to be in the range (0,1), in order to have true exponential decay. If set to 1, there is no decay, if set to 0, there is no memory. See py:meth:`~.TimeFrameAccumulator.accumulate` for details.
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
        """Sets the :py:meth:`~.TimeFrame.state` of the :py:meth:`~.TimeFrameAccumulator.time_frame` of self equal to the weighted sum of 
        the state of the new `time_frame` and the state the current time frame of self. The weight for the old state is 
        w = :py:meth:`~.TimeFrameAccumulator.decay_rate`^dt, where dt is the time of the provided `time_frame` minus the time-frame currently 
        held by self. The weight for the new `time_frame` is simply equal to 1.
        This method also sets the :py:meth:`~.TimeFrame.time_point` of the time-frame of self equal to that of the new `time_frame`.

        :param time_frame: The new time-frame to be added to the :py:meth:`~.TimeFrameAccumulator.time_frame` of self.
        :type time_frame: :py:class:`.TimeFrame`
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
        """Provides a :py:class:`.TimeFrame` that holds the time-discounted sum of all :py:class:`.TimeFrame` objects added via the :py:meth:`~.TimeFrameAccumulator.accumulate` method.

        :param current_time: The current time, used to discount the state of self.
        :type current_time: float
        :raises ValueError: If `current_time` is earlier than the time-point of the current time-frame of self.
        :return: The time-discounted time-frame of this accumulator.
        :rtype: :py:class:`.TimeFrame`
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
        """Resets the :py:meth:`~.TimeFrameAccumulator.time_frame` of self. If `initial_time_frame` is provided, then this one will
        be used for reset and saved in :py:meth:`~.TimeFrameAccumulator.initial_time_frame`. Otherwise, the one provided during construction will be used.

        :param initial_time_frame: The time-frame to be used to set :py:meth:`~.TimeFrameAccumulator.time_frame` and :py:meth:`~.TimeFrameAccumulator.initial_time_frame` of self.
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
        
class Connection(torch.nn.Module):
    """A connection between two :py:class:`Area` objects. This is analogous to a neural tract between areas of a biological neural network that 
    not only sends information but also converts it between the reference frames of the input and output area. It thus has a 
    :py:meth:`~.Connection.transformation` that is applied to the input before it is sent to the target area. For biological plausibility, 
    the transformation should be a simple linear transformation, for instance a :py:class:`torch.nn.Linear` layer.
    
    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param from_area_index: Sets the :py:meth:`~Connection.from_area_index` of this connection. 
    :type from_area_index: int
    :param to_area_index: Sets the :py:meth:`~Connection.to_area_index` of this connection. 
    :type to_area_index: int
    :param input_time_frame_accumulator: Used to set :py:meth:`~.Connection.input_time_frame_accumulator` of self.
    :type input_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param transformation: Sets the :py:meth:`~.Connection.transformation` of the connection.
    :type transformation: torch.nn.Module, optional, defaults to :py:class:`torch.nn.Identity`
    :return: A new connection.
    :rtype: :py:class:`.Connection`
    """

    def __init__(self, index: int, from_area_index: int, to_area_index: int, input_time_frame_accumulator: TimeFrameAccumulator, transformation: torch.nn.Module = torch.nn.Identity()) -> "Connection":
        
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
        :rtype: :py:class:`.TimeFrameAccumulator`
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
        """Reads the current state of the :py:meth:`.Connection.time_frame_accumulator` and applies the :py:meth:`~.Connection.transformation` to it. 

        :param current_time: The current time in the simulation.
        :type current_time: float  
        :return: The produced time frame.
        :rtype: :py:class:`.TimeFrame`
        """

        # Read input
        input_state = self.input_time_frame_accumulator.time_frame(current_time=current_time).state

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
    """An area corresponds to a small population of neurons that jointly hold a representation in the area's :py:meth:`~.Area.output_time_frame_accumulator`.
    Given a time-point t and a set S of areas that should be updated at t, the caller should update the areas' states in two consecutive loops over S. The first loop
    should call the :py:meth:`~.Area.collect_inputs` method on each area to make it collect, sum and buffer its inputs from the overall network. Then, in the second loop, the 
    :py:meth:`~Area.forward` method should be called on each area of S to sum the buffered inputs and apply the area's :py:meth:`~.Area.transformation`. 
    This splitting of input collection and forward transformation allows for parallelization of areas.
    
    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :raises ValueError: If the index is not a non-negative integer.
    :param output_time_frame_accumulator: Sets the :py:meth:`~.Area.output_time_frame_accumulator` of self.
    :type output_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param input_connections: Sets the :py:meth:`~.Area.input_connections` of this area.
    :type input_connections: List[:py:class:`.Connection`]
    :param input_shape: Sets the :py:meth:`~.Area.input_shape` of this area.
    :type input_shape: List[int]
    :param output_shape: Sets the :py:meth:`~.Area.output_shape` of this area.
    :type output_shape: List[int]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: List[:py:class:`.Connection`]
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area. If a st
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :return: A new area.
    :rtype: :py:class:`.Area`
    """

    def __init__(self, index: int, 
                 output_time_frame_accumulator: TimeFrameAccumulator, 
                 input_connections: List[Connection], 
                 input_shape: List[int],
                 output_shape: List[int],
                 output_connections: List[Connection],
                 transformation: torch.nn.Module, 
                 update_rate: float) -> "Area":
        
        # Call the parent constructor
        super().__init__()
        
        # Ensure input validity
        if not isinstance(input_shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            raise TypeError(f"The input_shape of area {index} must be a list of positive integers but was {input_shape}.")
        
        if not isinstance(output_shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in output_shape):
            raise TypeError(f"The output_shape of area {index} must be a list of positive integers but was {output_shape}.")
        if not output_shape == list(output_time_frame_accumulator._time_frame.state.shape[1:]):
            raise ValueError(f"The output_shape of area {index} must match the shape of the state of its output_time_frame_accumulator but was {output_shape} and {output_time_frame_accumulator.time_frame.state.shape[1:].as_list()}, respectively.")
        
        # Set properties
        self.index = index # Must be set first
        self.output_time_frame_accumulator = output_time_frame_accumulator
        self.input_connections = input_connections
        self._input_shape = input_shape
        self._output_shape = output_shape
        self.output_connections = output_connections
        
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
        """:return: The time-frame accumulator of this area. This holds the output state of the area which will be made available to other areas via :py:class:`.Connection`.
        :rtype: :py:class:`.TimeFrameAccumulator`"""
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
        """:return: A set of :py:class:`.Connection` objects projecting to this area.
        :rtype: Set[Connection]
        """
        return self._input_connections

    @input_connections.setter
    def input_connections(self, new_value: Set[Connection]) -> None:
        # Check input validity
        if not isinstance(new_value, Set):
            raise TypeError(f"The input_connections for area {self.index} must be a set of :py:class:`.Connection` objects projecting to area {self.index}.")
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
        """:return: The shape of the output of this area for a single instance (i.e. excluding the batch-dimension that is assumed to be at index 0 of the actual output). The output is the state held in the :py:meth:`~.Area.output_time_frame_accumulator` and hence has same shape.
        :rtype: int
        """
        return self._output_shape

    @property
    def output_connections(self) -> Set[Connection]:
        """:return: A set of :py:class:`.Connection` objects projecting from this area. 
        :rtype: Set[Connection]
        """
        return self._output_connections

    @output_connections.setter
    def output_connections(self, new_value: Set[Connection]) -> None:

        # Check input validity
        if not isinstance(new_value, Set):
            raise TypeError(f"The output_connections for area {self.index} must be a set of :py:class:`.Connection` objects projecting from area {self.index}.")
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
    def update_rate(self) -> float:
        """:return: The update-rate of this area.
        :rtype: float"""
        return self._update_rate
    
    @update_rate.setter
    def update_rate(self, new_value: float) -> None:

        # Check input validity
        if not isinstance(new_value, float):
            raise TypeError(f"The update_rate of area {self.index} has to be a float.")
        if not new_value > 0:
            raise ValueError(f"The update_rate of area {self.index} has to be positive.")
        
        # Set property
        self._update_rate = new_value

    @property
    def update_count(self) -> int:
        """:return: Counts how many times this area was updated during the simulation.
        :rtype: int"""
        return self._update_count
    
    def collect_inputs(self, current_time: float) -> None:
        """Calls the :py:meth:`~.Connection.forward` method of all incoming connections to get the current inputs, sums them up and buffers 
        the result for later use by the :py:meth:`~.Area.forward` method. Since the inputs are summed, it is necessary that they are all of the same shape. 
        
        :param current_time: The current time of the simulation used to time-discount the states of the input areas.
        :type current_time: float
        :rtype: None
        """

        # Initalize input state
        self._input_state = None
        
        # Accumulate inputs
        tmp = list(self.input_connections)
        if len(tmp) > 0: self._input_state = tmp[0].forward(current_time=current_time).state
        for connection in tmp[1:]: self._input_state += connection.forward(current_time=current_time).state

    def forward(self) -> None:
        """Assuming :py:meth:`~.Area.collect_inputs` has been run on all areas of the simulation just beforehand, this method passes the buffered inputs through
        the `:py:meth:`~.Area.transformation` of self (if exists) and passes the result to the :py:meth:`.TimeFrameAccumulator.accumulate` of self.
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

class Port(torch.nn.Module):

    def __init__(self, area_index: int) -> "Port":
        super().__init__()
        self._area_index = area_index

    @property
    def area_index(self) -> int:
        return self._area_index
    
    @property
    def input_signal_shape(self) -> List[int]:
        raise NotImplementedError("The input_signal_shape property must be implemented by subclasses of Port.")
    
    @property
    def output_signal_shape(self) -> List[int]:
        raise NotImplementedError("The output_signal_shape property must be implemented by subclasses of Port.")
    
    @property
    def modulator_shape(self) -> List[int]:
        raise NotImplementedError("The modulator_shape property must be implemented by subclasses of Port.")
    
    def forward(self, input_signal: torch.Tensor, modulator: torch.Tensor = None) -> torch.Tensor:
        if modulator == None:
            modulator = torch.ones((input_signal.shape[0], *self.modulator_shape), device=input_signal.device, dtype=input_signal.dtype)

        
    

class AreaTransformation(torch.nn.Module):

    class LinearRecursive(torch.nn.Module):

        def __init__(self, area_input_dimensionality, area_output_dimensionality) -> "AreaTransformation.LinearRecursive":
            super().__init__()
            self.area_input_dimensionality = area_input_dimensionality
            self.area_output_dimensionality = area_output_dimensionality

            self.A = torch.nn.Parameter(torch.randn((area_output_dimensionality, area_output_dimensionality))) # Transforms recurrent part
            self.B = torch.nn.Parameter(torch.randn((area_output_dimensionality, area_input_dimensionality - area_output_dimensionality))) # Transform the input from other areas
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:

            # Seperate recurrent part and other area input part
            x_self = x[:, :self.area_output_dimensionality]
            x_other = x[:, self.area_output_dimensionality:]

            # Transform inputs
            y_self = torch.matmul(x_self, self.A.T)
            y_other = torch.matmul(x_other, self.B.T)
            y = y_self + y_other

            # Output
            return y


class ConnectionTransformation(torch.nn.Module):

    class MoveToSlot(torch.nn.Module):

        def __init__(self, from_area_output_dimensionality, to_area_input_dimensionality, to_area_output_dimensionality, slot: str) -> "AreaTransformation.LinearRecursive":
            # Call super constructor
            super().__init__()
            
            # Validate input
            if not slot in ["self", "other"]:
                raise ValueError(f"The slot must be either 'self' or 'other', but was set to '{slot}'.")
            if not self.from_area_output_dimensionality == self.to_area_output_dimensionality:
                    raise ValueError(f"When moving to the 'self' slot, the from_area_output_dimensionality must be equal to the to_area_output_dimensionality, but they are {self.from_area_output_dimensionality} and {self.to_area_output_dimensionality}, respectively.")
            if not self.from_area_output_dimensionality < self.to_area_input_dimensionality:
                raise ValueError(f"The from_area_output_dimensionality must be smaller than the to_area_input_dimensionality, but they are {self.from_area_output_dimensionality} and {self.to_area_input_dimensionality}, respectively.")
            
            # Set properties            
            self.from_area_output_dimensionality = from_area_output_dimensionality
            self.to_area_input_dimensionality = to_area_input_dimensionality
            self.to_area_output_dimensionality = to_area_output_dimensionality
            self.slot = slot
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:

            # Prepare output
            batch_size = x.shape[0]
            y = torch.zeros((batch_size, self.to_area_input_dimensionality), device=x.device, dtype=x.dtype)
            
            # Move to slot
            if self.slot == "self":
                y[:, :self.to_area_output_dimensionality] = x
            else: # Other slot
                y[:, self.to_area_output_dimensionality:] = x
            
            # Output
            return y
        
class Source(Area):
    """The source :py:class:`.Area` is a special area because it streams the input to the other areas. In order to set it up for the simulation of a trial,
    load stimuli via the :py:meth:`~.Source.load_stimulus_batch method. Then, during each call to the :py:meth:`.~Area.collect_inputs` method, one :py:class:`.TimeFrame` 
    will be taken from the stimuli and held in a bffer. Upon calling the :py:meth:`~Area.forward` method, that time-frame will be placed in the
    :py:meth:`~.Area.TimeFrameAccumulator`, so that it can be read by other areas. Once the time frames are all streamed, the source area will no longer add new
    time-frames to the accumulator and hence its representation will simply decay over time.

    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param output_time_frame_accumulator: Sets the :py:meth:`~.Area.time_frame_accumulator` of this area. 
    :type output_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param output_shape: Sets the :py:meth:`~.Area.output_shape` of this area.
    :type output_shape: List[int]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`.Connection`]
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :return: An instance of this class.
    :rtype: :py:class:`.Source`
    """

    def __init__(self, index: int, output_time_frame_accumulator: TimeFrameAccumulator, output_shape: List[int], output_connections: Dict[int, Connection], update_rate: float, data_loader: torch.utils.data.DataLoader=None) -> "Source":

        # Call the parent constructor
        super().__init__(index=index,
                         output_time_frame_accumulator=output_time_frame_accumulator,
                         input_connections=set([]),
                         input_shape=[],
                         output_shape=output_shape,
                         output_connections=output_connections,
                         transformation=torch.nn.Identity(),
                         update_rate=update_rate)
        
        # Set properties
        self.data_loader = data_loader
        self._stimulus_batch = None

    @property
    def data_loader(self) -> torch.utils.data.DataLoader:
        """The dataloader used to fetch data in batches before streaming it time-frame by time-frame to other areas. It is assumed that the batches produced by the dataloader are torch.Tensors of shape [batch size, time-frame count, ...], where ... is the shape of an individual stimulus time-frame.
        
        :return: The data loader.
        :rtype: torch.utils.data.DataLoader
        """
        return self._data_loader

    @data_loader.setter
    def data_loader(self, new_value) -> None:
        # Handle None
        if new_value == None: 
            self._data_loader = None
            return
        
        # Check input vaidity
        if not isinstance(new_value, torch.utils.data.DataLoader): raise TypeError(f"Expected data_loader to be of type torch.utils.data.DataLoader but received {type(new_value)}.")

        # Set
        self._data_loader = new_value

    @property
    def stimulus_batch(self) -> Deque[TimeFrame]:
        """The stimuli that are currently loaded in the source area. This is a deque of :py:class:`.TimeFrame` objects that are to be processed by the model.
        
        :return: The stimuli.
        :rtype: Deque[:py:class:`.TimeFrame`]
        """
        return self._stimulus_batch

    def load_next_stimulus_batch(self) -> None:
        """This method loads the next batch of stimuli that will be streamed to the other model areas during the simulation. 

        :raises Exception: if self.data_loader is None.
        :raises StopIteration: if the data_loader is empty.
        """
        
        # Check input validity
        if self.data_loader == None: raise Exception(f"Unable to load the next stimulus batch for source {self.index} because the data_loader is None.")
        X, y = next(iter(self.data_loader))
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
        """Pops the next :py:class:`.TimeFrame` from :py:meth:`~.Source.stimulus_batch` or generates an array of zeros if the stimulus stream is over. Either way, the result is buffered internally to be made available to other areas upon calling :py:meth:`~.Area.forward`.
        
        :param current_time: The current time of the simulation.
        :type current_time: float
        :raises ValueError: if the `current_time` is not equal to the time of the popped :py:class:`.TimeFrame`.
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
    """This class is a subclass of :py:class:`.Area` and has the same functionality as a regular area except that it has no output connections.

    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param output_time_frame_accumulator: Sets the :py:meth:`~.Area.output_time_frame_accumulator` of self.
    :type output_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param input_connections: Sets the :py:meth:`~.Area.input_connections` of this area.
    :type input_connections: List[:py:class:`.Connection`]
    :param input_shape: Sets the :py:meth:`~.Area.input_shape` of this area.
    :type input_shape: List[int]
    :param output_shape: Sets the :py:meth:`~.Area.output_shape` of this area.
    :type output_shape: List[int]
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    """

    def __init__(self, index: int, 
                 output_time_frame_accumulator: TimeFrameAccumulator, 
                 input_connections: List[Connection],
                 input_shape: List[int],
                 output_shape: List[int],
                 transformation: torch.nn.Module, 
                 update_rate: float) -> "Target":
        super().__init__(index=index, 
                 output_time_frame_accumulator=output_time_frame_accumulator, 
                 input_connections=input_connections, 
                 input_shape=input_shape,
                 output_shape = output_shape,
                 output_connections=None,
                 transformation=transformation, 
                 update_rate=update_rate)
     
    @Area.output_connections.setter
    def output_connections(self, new_value: List[Connection]) -> None:
        if new_value != None:
            raise ValueError("A Target area does not accept any output connections.")

class BrIANN(torch.nn.Module):
    """This class functions as the network that holds together all its :py:class:`.Area`s and :py:class:`.Connection`s. Its name abbreviates Brain Inspired Artificial Neural Networks. 
    To use it, one should provide a configuration dictionary from which all components can be loaded. 
    Then, for each batch, one should call :py:meth:`~.BrIANN.load_next_stimulus_batch`.
    Once a batch is loaded, the processing can be simulated for as long as the caller intends (ideally at least for as long as the
    :py:class:`~.Source` areas provide :py:class:`.TimeFrame`s) using the :py:meth:`~.BrIANN.step` method.
    In order to get a simplified networkx representation which contains information about the large-scale network topology (:py:class:`.Area`s and :py:class:`.Connection`s),
    one can use :py:meth:`~.BrIANN.get_topology`.
    
    :param configuration: A configuration in the form of a dictionary.
    :type configuration: Dict[str, Any]
    """
    
    def __init__(self, configuration: Dict[str,Any]) -> "BrIANN":
        
        # Call the parent constructor
        super().__init__()

        # Load the model based on the configuration
        self._load_from_configuration(configuration=configuration)

        # Set the time of the simulation
        self._current_simulation_time = 0.0
        """:return: The time that has passed since the start of the simulation. It is updated after each step of the simulation.
        :rtype: float
        """

    @property
    def areas(self) -> Set[Area]:
        """:return: The set of areas held by self.
        :rtype: Set[:py:class:`.Area`]
        """

        return self._areas

    def get_area_indices(self) -> Set[int]:
        """:return: The set of indices of the areas stored internally.
        :rtype: Set[int]
        """
        return set([area.index for area in self._areas])
    
    def get_area_at_index(self, index: int) -> Area:
        """:return: The area with given `index`.
        :rtype: :py:class:`.Area`
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
    def connections(self) -> Set[Connection]:
        """:return: The set of internally stored :py:class:`.Connection`.
        :rtype: Set[:py:class:`.Connection`]
        """
        return self._connections
    
    def get_connections_from(self, area_index: int) -> Set[Connection]:
        """:return: A set of :py:class:`.Connection` objects that are the output connections of the area with the given index. 
        :rtype: Set[:py:class:`.Connection`]
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
        """:return: A set of :py:class:`.Connection` objects that are the input connections to the area with the given index. 
        :rtype: Set[:py:class:`.Connection`]
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

    def _load_from_configuration(self, configuration: Dict[str, Any]) -> None:
        """Loads the overall network, including the :py:class:`.Area`s and :py:class:`.Connection`s as well as
        the torch.data.utils.DataLoaders.
        
        :param configuration: A configuration in the form of a dictionary.
        :type configuration: Dict[str, Any]
        """
        
        # Check if all area indices are integers 
        area_indices = [item["index"] for item in configuration["areas"]]
        if not all(isinstance(area_index, int) for area_index in area_indices):
            raise TypeError("All area indices must be integers.")
        
        # Extract network parameters
        network_configuration = configuration["network"]
        name = network_configuration["name"]
        decay_rate = network_configuration["decay_rate"]
        batch_size = network_configuration["batch_size"]

        # Ensure their validity
        if not isinstance(name, str):
            raise TypeError(f"The network's name was expected to be a str, but is {type(name)}")
        if (not isinstance(decay_rate, int)) and (not isinstance(decay_rate, float)):
            raise TypeError(f"The network's decay_rate was expected to be a float, but is {type(decay_rate)}")
        decay_rate = (float)(decay_rate)
        if not isinstance(batch_size, int):
            raise TypeError(f"The network's batch_size was expected to be an int, but is {type(batch_size)}")
        if not batch_size > 0:
            raise TypeError(f"The network's batch_size was expected to be positive, but is {batch_size}")
        
        # Set properties
        self.name = name
        self.decay_rate = decay_rate
        self.batch_size = batch_size

        # Extract initial states from areas to pass to connections
        time_frame_accumulators = {}
        for area_configuration in configuration["areas"]:
            
            # Extract initial state
            global initial_state
            exec("global initial_state; initial_state = " + area_configuration["initial_state"]) # This is a single tensor that is disregarding the batch-size
            
            if not isinstance(initial_state, torch.Tensor):
                raise TypeError(f"Expected initial_state to be a torch.Tensor, but received {type(initial_state)}.")
            
            # Create copies for all instances of a batch
            initial_state = torch.concatenate([initial_state[torch.newaxis, :] for _ in range(batch_size)], dim=0)

            # Create time frame accumulator
            time_frame = TimeFrame(state=initial_state, time_point=0.0)
            time_frame_accumulator = TimeFrameAccumulator(initial_time_frame=time_frame, decay_rate=decay_rate) 
            
            # Store
            index = area_configuration["index"]
            time_frame_accumulators[index] = time_frame_accumulator

        # Set connections
        self._connections = set([])

        for connection_configuration in configuration["connections"]:
            # Extract configuration
            index = connection_configuration["index"]
            from_area_index = connection_configuration["from_area_index"]
            to_area_index = connection_configuration["to_area_index"]
            time_frame_accumulator = time_frame_accumulators[from_area_index]
            global transformation
            exec("global transformation; transformation = " + connection_configuration["transformation"])
            connection = Connection(index=index, from_area_index=from_area_index, to_area_index=to_area_index, input_time_frame_accumulator=time_frame_accumulator, transformation=transformation)
            
            # Insert the connection to the arrays
            self._connections.add(connection)
            
        # Set areas
        self._areas = set([])
        for area_configuration in configuration["areas"]:
            
            # Enrich configuration
            area_index = area_configuration["index"]

            if area_configuration["type"] != "Source": area_configuration["input_connections"] = self.get_connections_to(area_index=area_index)
            area_configuration["output_time_frame_accumulator"] = time_frame_accumulators[area_index]
            if area_configuration["type"] != "Target": area_configuration["output_connections"] = self.get_connections_from(area_index=area_index)

            if "hold_function" in area_configuration.keys(): 
                global hold_function
                exec("global hold_function; hold_function = " + area_configuration["hold_function"])
                area_configuration["hold_function"] = hold_function

            if "transformation" in area_configuration.keys():
                exec("global transformation; transformation = " + area_configuration["transformation"])
                area_configuration["transformation"] = transformation

            if "dataset_index" in area_configuration.keys():
                dataset_index = area_configuration["dataset_index"]
                global dataset_configuration
                dataset_configuration = configuration["datasets"][dataset_index]
                dataset_type = dataset_configuration["type"]
                del dataset_configuration["type"]
                del dataset_configuration["index"]

                try:
                    # Create data_loader
                    global dataset
                    exec("global dataset, dataset_configuration; dataset = bptdm." + dataset_type + "(**dataset_configuration)")
                    area_configuration["data_loader"] = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
                except:
                    pass

                # Remove other attributes
                del area_configuration["dataset_index"]

            # Create area
            area_type = area_configuration["type"]
            del area_configuration["type"]
            del area_configuration["initial_state"]
            global tmp, area
            tmp = area_configuration
            exec("global area, tmp; area = " + area_type + "(**tmp)")

            # Store
            self._areas.add(area)
           
    def get_topology(self) -> nx.DiGraph:
        """Converts the BrIANN network to a NetworkX DiGraph where each node is simply the :py:meth:`~.Area.index` of a corresponding :py:class:`.Area`
        and each edge is simply the triplet (*u*,*v*) where *u* is the :py:meht:`~Connection.from_index`, *v* the :py:meht:`~Connection.to_index` of the corresponding :py:class:`.Connection`.
        
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

    def load_next_stimulus_batch(self) -> None:
        """This method resets the :py:meth:`.~BrIANN.current_simulation_time` and all areas. It also makes the :py:class:`.Source` areas
        load their corresponding next batch of stimuli. It thus assumes that all source areas have a valid :py:meth:`~.Source.data_loader` set
        and that the data loaders are in sync with each other and non-empty.

        :raises StopIteration: If a :py:meth:`.Source.data_loader` reached its end.
        :rtype: None
        """
        
        # Reset the states of all areas
        for area in self.areas:
            area.reset()

        # Load the next batch of stimuli into the source areas
        for area in self.areas:
            if isinstance(area, Source):
                area.load_next_stimulus_batch()

        # Reset the simulation time
        self._current_simulation_time = 0.0

    def step(self) -> Set[Area]:
        """Performs one step of the simulation by finding the set of areas due to be updated next and calling their :py:meth:`~.Area.collect_inputs` and
        :py:meth:`~.Area.forward` method to make them process their inputs. 
        This method needs to be called repeatedly to step through the simulation. The simulation does not have an internally checked stopping condition,
        meaning this step method can be called indefinitely, even if the sources already ran out of stimuli. 
        The caller of this method thus needs to determine when to stop the simulation.

        :return: The set of areas that were updated within this step.
        :rtype: Set[:py:class:`~.Area`]
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

    def __repr__(self) -> str:
        string = "BrIANN\n"

        for area in self._areas: 
            string += f"{area}\n"
            for connection in self.get_connections_from(area_index=area.index):
                string += f"\t{connection}\n"

        return string


