"This module collects all necessary components to build a BrIANN model."
import numpy as np
import torch
from typing import List, Dict, Deque, Set, Any

import sys
from abc import ABC, abstractmethod
import json
import matplotlib.pyplot as plt
import os
sys.path.append(os.path.abspath(""))
from src.briann.python.training import data_management as dmg
import networkx as nx

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
            raise ValueError(f"When reading a TimeFrame, the provided current_time ({current_time}) must be later than that of the time-frame held by self ({self._time_frame.time_point}).")
        
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
    should call the :py:meth:`~.Area.collect_inputs` method on each area to make it buffer its inputs from the overall network. Then, in the second loop, the 
    :py:meth:`~Area.forward` method should be called on each area of S to apply its corresponding :py:meth:`~Area.state_merge_strategy` and 
    :py:meth:`~.Area.transformation` to the buffered inputs. This splitting of input collection and forward transformation allows for parallelization of areas.
    
    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :raises ValueError: If the index is not a non-negative integer.
    :param output_time_frame_accumulator: Sets the :py:meth:`~.Area.output_time_frame_accumulator` of self.
    :type output_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param input_connections: Sets the :py:meth:`~.Area.input_connections` of this area.
    :type input_connections: List[:py:class:`.Connection`]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: List[:py:class:`.Connection`]
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area. If a st
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :param state_merge_strategy: Sets the :py:meth:`~.Area.state_merge_strategy` of this area.
    :type state_merge_strategy: callable, optional, defaults to None
    :return: A new area.
    :rtype: :py:class:`.Area`
    """

    def __init__(self, index: int, 
                 output_time_frame_accumulator: TimeFrameAccumulator, 
                 input_connections: List[Connection], 
                 output_connections: List[Connection],
                 transformation: torch.nn.Module, 
                 update_rate: float, 
                 state_merge_strategy: callable = None) -> "Area":
        
        # Call the parent constructor
        super().__init__()
        
        # Set properties
        self.index = index # Must be set first
        self.output_time_frame_accumulator = output_time_frame_accumulator
        self.input_connections = input_connections
        self.output_connections = output_connections
        
        # Check input validity
        if not isinstance(transformation, torch.nn.Module):
            raise TypeError(f"The transformation of area {self.index} must be a torch.nn.Module object.")
        
        self._transformation = transformation # With torch, it is not possible to use the regular property setter/ getter, hence, the transformation is set once here manually and then kept private
        
        self.update_rate = update_rate
        self._update_count = 0
        self.state_merge_strategy = state_merge_strategy
        self._input_states = None # Will store the buffered input states updated by collect_inputs

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
    def input_connections(self) -> List[Connection]:
        """:return: A list of :py:class:`.Connection` objects projecting to this area.
        :rtype: List[Connection]
        """
        return self._input_connections

    @input_connections.setter
    def input_connections(self, new_value: List[Connection]) -> None:
        # Check input validity
        if not isinstance(new_value, List):
            raise TypeError(f"The input_connections for area {self.index} must be a list of :py:class:`.Connection` objects projecting to area {self.index}.")
        if not all(isinstance(connection, Connection) for connection in new_value):
            raise TypeError(f"All values in the input_connections list of area {self.index} must be Connection objects projecting to area {self.index}.")
        
        # Set property
        self._input_connections = new_value 

    @property
    def output_connections(self) -> List[Connection]:
        """:return: A list of :py:class:`.Connection` objects projecting from this area. 
        :rtype: List[Connection]
        """
        return self._output_connections

    @output_connections.setter
    def output_connections(self, new_value: List[Connection]) -> None:

        # Check input validity
        if not isinstance(new_value, List):
            raise TypeError(f"The output_connections for area {self.index} must be a list of :py:class:`.Connection` objects projecting from area {self.index}.")
        if not all(isinstance(connection, Connection) for connection in new_value):
            raise TypeError(f"All values in the output_connections list of area {self.index} must be Connection objects projecting from area {self.index}.")
        if 0 < len(new_value):
            time_frame_accumulator = new_value[0].input_time_frame_accumulator
            for connection in new_value[1:]:
                if not connection.input_time_frame_accumulator == time_frame_accumulator:
                    raise ValueError("When setting the output_connections of an area, they must all have the same input_time_frame_accumulator")  

        # Set property
        self._output_connections = new_value

        # Set output_time_frame_accumulator
        if 0 < len(new_value):
            self._output_time_frame_accumulator = new_value[0].input_time_frame_accumulator
    
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
    
    @property
    def state_merge_strategy(self) -> callable:
        """:return: The state-merge-strategy of this area. This is a function mapping from a <input area index, state tensor> dictionary to a single torch.nn.Tensor. It is used to combine the inputs to this area during :py:meth:`~.Area.collect_inputs`. If set to None, this function will not be used and the transformation of self needs to be able to process the dictionary of inputs. :py:class:`.Source` objects must not have a state-merge-strategy because they have no inputs.
        :rtype: callable"""
        return self._state_merge_strategy

    @state_merge_strategy.setter
    def state_merge_strategy(self, new_value: callable) -> None:
        
        # Ensure input validity
        if new_value != None:
            if not callable(new_value):
                raise TypeError(f"The state_merge_strategy was expected to be a callable but is a {type(new_value)}.")
        
        # Set property
        self._state_merge_strategy = new_value

    def collect_inputs(self, current_time: float) -> None:
        """Calls the :py:meth:`~.Connection.forward` method of all incoming connections to get the current inputs, applies the :py:meth:`~.Area.state_merge_strategy`
        of self (if not None) and buffers the result for later use by the :py:meth:`~.Area.forward` method.
        
        :param current_time: The current time of the simulation used to time-discount the states of the input areas.
        :type current_time: float
        :rtype: None
        """

        # Collect all inputs
        self._input_states = {}
        for i, connection in enumerate(self.input_connections):
           self._input_states[i] = connection.forward(current_time=current_time).state
            
        # Apply merge strategy
        if self._state_merge_strategy != None: self._input_states = self._state_merge_strategy(self._input_states)

    def forward(self) -> None:
        """Assuming :py:meth:`~.Area.collect_inputs` has been run on all areas of the simulation just beforehand, this method passes the buffered inputs through
        the `:py:meth:`~.Area.transformation` of self and passes the result to the :py:meth:`.TimeFrameAccumulator.accumulate` of self.
        """

        # Determine current time
        self._update_count += 1
        current_time = self._update_count / self.update_rate

        # Retrieve inputs
        if self._input_states == None:
            raise ValueError(f"The input_states of area {self.index} are None. Please run collect_inputs() on all areas before calling forward().")
        input_states = self._input_states
        self._input_states = None

        # Apply transformation to the states
        new_state = self._transformation.forward(input_states)

        # Create and accumulate a new time frame for the current state
        new_time_frame = TimeFrame(state=new_state, time_point=current_time)
        self._output_time_frame_accumulator.accumulate(time_frame=new_time_frame)        

        # Notify subscribers
        if hasattr(self, "_subscribers"):
            for subscriber in self._subscribers:
                subscriber.receive_state(area_index=self.index, time_frame=new_time_frame)

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
                subscriber.receive_state(area_index=self.index, time_frame=new_time_frame)

    
    def add_state_subscriber(self, subscriber: Any) -> None:
        """Adds a subscriber to the area. The subscriber must have a method `receive_state(area_index: int, time_frame: TimeFrame)` that will be called every time the area is updated.

        :param subscriber: The subscriber to be added.
        :type subscriber: Any
        """

        if not hasattr(subscriber, "receive_state"):
            raise ValueError("The subscriber must have a method receive_state(area_index: int, time_frame: TimeFrame).")
        
        if not callable(subscriber.receive_state):
            raise ValueError("The receive_state attribute of the subscriber must be callable.")
        
        if not len(subscriber.receive_state.__code__.co_varnames) == 3:
            raise ValueError("The receive_state method of the subscriber must have exactly two parameters: area_index: int and time_frame: TimeFrame.")
        
        if not subscriber.receive_state.__code__.co_varnames[1] == "area_index":
            raise ValueError("The first parameter of the receive_state method of the subscriber must be named area_index.")
        
        if not subscriber.receive_state.__code__.co_varnames[2] == "time_frame":
            raise ValueError("The second parameter of the receive_state method of the subscriber must be named time_frame.")
        
        if not subscriber.receive_state.__code__.co_argcount == 3:
            raise ValueError("The receive_state method of the subscriber must have exactly two parameters: area_index: int and time_frame: TimeFrame.")
        
        if not hasattr(self, "_subscribers"):
            self._subscribers = []
        
        self._subscribers.append(subscriber)

    def __repr__(self) -> str:
        """Returns a string representation of the area."""
        return f"Area(index={self._index}, update_rate={self._update_rate}, update_count={self._update_count})"

class AreaStateSubscriber(ABC):
    """An abstract base class for subscribers that want to receive the state of an area every time it is updated. Subscribers must implement the :py:meth:`~.AreaStateSubscriber.receive_state` method.
    """

    @abstractmethod
    def receive_state(self, area_index: int, time_frame: TimeFrame) -> None:
        """This method will be called every time the area is updated. The subscriber can then process the received state as desired.

        :param area_index: The index of the area that was updated.
        :type area_index: int
        :param time_frame: The time frame that was produced by the area.
        :type time_frame: TimeFrame
        """
        pass

class Source(Area):
    """The source :py:class:`.Area` is a special area because it streams the input to the other areas. In order to set it up for the simulation of a trial,
    load stimuli via the :py:meth:`~.Source.load_stimulus_batch method. Then, during each call to the :py:meth:`.~Area.forward` method, one :py:class:`.TimeFrame` 
    will be taken from the stimuli and placed in the :py:meth:`~.Area.TimeFrameAccumulator` other areas. Once the time frames are all streamed, the source area will use the **hold_function**
    to hold the last time frame for a specified **cool_down_duration**, to let the other areas finish their processing.

    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param output_time_frame_accumulator: Sets the :py:meth:`~.Area.time_frame_accumulator` of this area. 
    :type output_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`.Connection`]
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :param cool_down_duration: Sets the :py:meth`.~.Source.cool_down_duration` of the source area.
    :type cool_down_duration: float
    :param hold_function: Sets the :py:meth:`~.Source.hold_function` of the source area.
    :type hold_function: Callable, optional, default to lambda last_state: torch.zeros_like(last_state)
    """

    def __init__(self, index: int, output_time_frame_accumulator: TimeFrameAccumulator, output_connections: Dict[int, Connection], update_rate: float, cool_down_duration: float, hold_function: callable = lambda last_state: torch.zeros_like(last_state), data_loader: torch.utils.data.DataLoader=None) -> "Source":

        # Call the parent constructor
        super().__init__(index=index,
                         output_time_frame_accumulator=output_time_frame_accumulator,
                         input_connections=[],
                         output_connections=output_connections,
                         transformation=torch.nn.Identity(),
                         update_rate=update_rate)
        
        # Set properties
        self.data_loader = data_loader
        self.cool_down_duration = cool_down_duration
        self.remaining_cool_down_duration = cool_down_duration
        self.hold_function = hold_function
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
    def cool_down_duration(self) -> float:
        """The cool down duration of the source area. This is the duration in seconds for which the source area shall hold the last state using :py:meth:`~.Source.hold_function` at the end of the simulation to let the other areas finish their processing.
        
        :return: The cool down duration.
        :rtype: float
        """
        return self._cool_down_duration
    
    @cool_down_duration.setter
    def cool_down_duration(self, new_value: float) -> None:
        if not isinstance(new_value, float):
            raise TypeError("The cool_down_duration must be a float.")
        if not new_value >= 0:
            raise ValueError("The cool_down_duration must be greater than or equal to 0.")
        self._cool_down_duration = new_value

    @property
    def remaining_cool_down_duration(self) -> float:
        """The remaining :py:meth:`~.Source.cool_down_duration` to be decayed once the current stimuli have been streamed. This is used to determine the end of the simulation.

        :return: The cool down duration.
        :rtype: float
        """
        return self._cool_down_duration
    
    @remaining_cool_down_duration.setter
    def remaining_cool_down_duration(self, new_value: float) -> None:
        if not isinstance(new_value, float):
            raise TypeError("The remaining_cool_down_duration must be a float.")
        if not new_value >= 0:
            raise ValueError("The remaining_cool_down_duration must be greater than or equal to 0.")
        self._remaining_cool_down_duration = new_value

    @property
    def hold_function(self) -> callable:
        """The hold function that is called whenever the source area has no more time frames to process. It is used to hold the state of the source areas last :py:class:`.TimeFrame` object for :py:meth:`~.Source.cool_down_duration` seconds while the remaining model areas are still processing the input.
        
        :return: The hold function.
        :rtype: callable
        """
        return self._hold_function

    @hold_function.setter
    def hold_function(self, new_value) -> None:
        if not callable(new_value):
            raise TypeError("The hold_function must be a Callable that takes as input a tensor which is equal to the state of the last TimeFrame object generated from the input to the simulation.")
        self._hold_function = new_value

    @Area.state_merge_strategy.setter
    def state_merge_strategy(self, new_value: callable) -> None:
        
        if new_value != None:
            raise ValueError("Sources must not have a state_merge_strategy")

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

        # Reset the remaining cool down duration
        self._remaining_cool_down_duration = self._cool_down_duration

        # Load first time-frame
        self._update_count = self.update_count - 1 # This will be incremented again in the forward method and then the first data point corresponds to the default update count
        self.forward()

    def collect_inputs(self, current_time: float) -> None:
        pass

    def forward(self) -> None:
        """Pops the next time frame from :py:meth:`~.Source.stimulus_batch` and passes it to the :py:meth:`~.Area.output_time_frame_accumulator`.
        If there are no more stimuli left, applies the :py:meth:`~.Source.hold_function` to a copy of the last stimulus :py:class:`.TimeFrame`."""

        # Increment update count
        self._update_count += 1

        # Get the next time frame
        if len (self._stimulus_batch) > 0:
            new_time_frame = self._stimulus_batch.pop()
        elif self._remaining_cool_down_duration > 0:
            # If there are no more time frames, hold the last state for the cool down duration
            # Continue the stream
            state = self._hold_function(self._last_time_frame.state)
            dt = 1/self.update_rate
            new_time_frame = TimeFrame(state=state, time_point = self._last_time_frame.time_point + dt)
            self._remaining_cool_down_duration -= self._processing_time
        else: # Simulation is over
            return    
        
        # Store a reference to the current time-frame for later
        self._last_time_frame = new_time_frame    
        
        # Pass the time frame to the accumulator
        self.output_time_frame_accumulator.accumulate(time_frame = new_time_frame)

        # Notify subscribers
        if hasattr(self, "_subscribers"):
            for subscriber in self._subscribers:
                subscriber.receive_state(area_index=self.index, time_frame=new_time_frame)

    def reset(self) -> None:

        # Call parent
        super().reset()

        # Reset remaining cool down duration
        self._remaining_cool_down_duration = self._cool_down_duration

class Target(Area):
    """This class is an subclass of :py:class:`.Area` and has the same functionality as a regular area except that it has no output connections.

    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param output_time_frame_accumulator: Sets the :py:meth:`~.Area.output_time_frame_accumulator` of self.
    :type output_time_frame_accumulator: :py:class:`.TimeFrameAccumulator`
    :param input_connections: Sets the :py:meth:`~.Area.input_connections` of this area.
    :type input_connections: List[:py:class:`.Connection`]
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :param state_merge_strategy: Sets the :py:meth:`~.Area.state_merge_strategy` of this area.
    :type state_merge_strategy: callable, optional, defaults to None
    """

    def __init__(self, index: int, 
                 output_time_frame_accumulator: TimeFrameAccumulator, 
                 input_connections: List[Connection],
                 transformation: torch.nn.Module, 
                 update_rate: float, 
                 state_merge_strategy: callable = None) -> "Target":
        super().__init__(index=index, 
                 output_time_frame_accumulator=output_time_frame_accumulator, 
                 input_connections=input_connections, 
                 output_connections=[],
                 transformation=transformation, 
                 update_rate=update_rate, 
                 state_merge_strategy=state_merge_strategy)
     
class BrIANN(torch.nn.Module):
    """This class functions as the network that holds together all its :py:class:`.Area`s and :py:class:`.Connection`s. Its name abbreviates Brain Inspired Artificial Neural Networks. 

    :param batch_size: The batch size used when passing instances through the areas.
    :type batch_size: int
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

        # Set next areas 
        self._due_areas = []

    @property
    def areas(self) -> Set[Area]:
        """The set of areas held by self.

        :return: The set of areas held by self.
        :rtype: Set[:py:class:`.Area`]
        """

        return self._areas

    def get_area_indices(self) -> List[int]:
        """Returns the list of indices of the areas stored internally.

        :return: The list of indices.
        :rtype: List[int]
        """
        return [area.index for area in self._areas]
    
    def get_area_at_index(self, index: int) -> Area:
        """Returns the area with given `index`.

        :return: The area with given `index`.
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
    def connections(self) -> Dict[int, Connection]:
        """A dictionary where each key is an area index and each value is a :py:class:`.Connection`.
        
        :return: The dictionary of :py:class:`.Connection`s .
        :rtype: Dict[int, :py:class:`.Connection`]
        """
        return self._connections
    
    def connections_from(self, area_index: int) -> List[Connection]:
        """A list of :py:class:`.Connection` objects that are the output connections of the area with the given index. 
        
        :return: The list of output connections.
        :rtype: List[:py:class:`.Connection`]
        """

        # Compile
        result = [None] * len(self._connections)
        i = 0
        for connection in self._connections:
            if connection.from_area_index == area_index: 
                result[i] = connection
                i += 1

        # Output
        return result[:i]
        
    def connections_to(self, area_index: int) -> List[Connection]:
        """A list of :py:class:`.Connection` objects that are the input connections to the area with the given index. 
        
        :return: The selistt of input connections.
        :rtype: List[:py:class:`.Connection`]
        """

        # Compile
        result = [None] * len(self._connections)
        i = 0
        for connection in self._connections:
            if connection.to_area_index == area_index: 
                result[i] = connection
                i += 1

        # Output
        return result[:i]
    
    @property
    def current_simulation_time(self) -> float:
        """The current simulation time in seconds. This is the time that has passed since the start of the simulation. It is updated after each step of the simulation.
        
        :return: The current simulation time.
        :rtype: float
        """
        return self._current_simulation_time

    def _load_from_configuration(self, configuration: Dict[str, Any]) -> None:
        """Loads the configuration from the given **json_string** and sets up the areas and connections.
        
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

            if area_configuration["type"] != "Source": area_configuration["input_connections"] = self.connections_to(area_index=area_index)
            area_configuration["output_time_frame_accumulator"] = time_frame_accumulators[area_index]
            if area_configuration["type"] != "Target": area_configuration["output_connections"] = self.connections_from(area_index=area_index)

            if "hold_function" in area_configuration.keys(): 
                global hold_function
                exec("global hold_function; hold_function = " + area_configuration["hold_function"])
                area_configuration["hold_function"] = hold_function

            if "transformation" in area_configuration.keys():
                exec("global transformation; transformation = " + area_configuration["transformation"])
                area_configuration["transformation"] = transformation

            if "state_merge_strategy" in area_configuration.keys():
                global state_merge_strategy
                exec("global state_merge_strategy; state_merge_strategy = " + area_configuration["state_merge_strategy"])
                area_configuration["state_merge_strategy"] = state_merge_strategy

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
                    exec("global dataset, dataset_configuration; dataset = dmg." + dataset_type + "(**dataset_configuration)")
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
            
        # Set flag to indicate that all states are reset
        self._all_areas_reset = True

    def to_simple_networkx(self) -> nx.DiGraph:
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
        
        # Reset the simulation time
        self._current_simulation_time = 0.0

        # Reset the states of all areas
        if not self._all_areas_reset:
            for area in self._areas:
                area.reset()
            self._all_areas_reset = True

        # Load the next batch of stimuli into the source areas
        for area in self.areas:
            if isinstance(area, Source):
                area.load_next_stimulus_batch()

    def step(self) -> None:
        """Performs one step of the simulation by processing the next :py:class:`.TimeFrame` from the source areas and passing it through the areas.
        This method needs to be called repeatedly until the end of the simulation.

        :raises StopIteration: if the simulation is over.
        
        """
        
        # If no more areas are due, stop the simulation
        for area in self.areas:
            if isinstance(area, Source) and area.remaining_cool_down_duration <= 0:            
                raise StopIteration("Cannot take another step because simulation is over.")

        # Find the areas that are due next
        self._due_areas = set([])
        min_time = sys.float_info.max
        for area in self._areas:
            area_next_time = (area.update_count +1) / area.update_rate # Add 1 to get the time of the area's next frame 
            if area_next_time == min_time: # Current area belongs to current set of due areas
                self._due_areas.add(area)
            elif area_next_time < min_time: # Current area is due sooner 
                self._due_areas = set([area])
                min_time = area_next_time
            
        # Update the simulation time
        self._current_simulation_time = min_time

        # Make all areas collect their inputs
        for area in self._due_areas: area.collect_inputs(current_time=self._current_simulation_time)

        # Make all areas process their inputs
        for area in self._due_areas: area.forward()

    def __repr__(self) -> str:
        string = "BrIANN\n"

        for area in self._areas.values(): 
            string += f"{area}\n"
            for connection in self._connection_from[area.index]:
                string += f"\t{connection}\n"

        return string
