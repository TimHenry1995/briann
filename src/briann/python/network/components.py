"THIS IS A TEST COMMEMNT"

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

from abc import ABC, abstractmethod
from collections import deque

class TimeFrame():
    """A time frame in the simulation that holds a temporary state of an :py:class:`.Area`. Note, this constructor automatically computes the :py:attr:`~.TimeFrame.end_time` based on the provided `start_time` and `duration`.

    :param state: Sets the :py:attr:`~.TimeFrame.state` of this time frame.
    :type state: :py:class:`torch.Tensor`
    :param index: Sets the :py:attr:`~.TimeFrame.index` of this time frame.
    :type index: int
    :param start_time: Sets the :py:attr:`~.TimeFrame.start_time` of this time frame.
    :type start_time: float
    :param duration: Sets the :py:attr:`~.TimeFrame.duration` of this time frame.
    :type duration: float
    """
    
    def __init__(self, state: torch.Tensor, index: int, start_time: float, duration: float) -> None:
        
        # Set state
        if not isinstance(state, torch.Tensor):
            raise TypeError("The state must be a torch.Tensor.")
        self._state = state
        
        # Set index
        if not isinstance(index, int):
            raise TypeError("The index must be an int.")
        if not index >= 0:
            raise ValueError("The index must be greater than or equal to 0.")
        self._index = index

        # Set times
        if not isinstance(start_time, float):
            raise TypeError("The start time must be a float.")
        self._start_time = start_time

        if not isinstance(duration, float):
            raise TypeError("The duration must be a float.")
        if not duration > 0:
            raise ValueError("The duration must be greater than 0.")
        self._duration = duration

        self._end_time = start_time + duration

    @property
    def state(self) -> torch.Tensor:
        """The state of the time frame. This is a :py:class:`torch.tensor` without an explicit time axis, for instance of shape [instance count, dimensionality]."""
        return self._state

    @property
    def index(self) -> int:
        """The index of the time frame. This is used to identify a timeframe."""
        return self._index
    
    @property
    def start_time(self) -> float:
        """The start time of the time frame."""
        return self._start_time
    
    @property
    def duration(self) -> float:
        """The duration of the time frame."""
        return self._duration

    @property
    def end_time(self) -> float:
        """The end time of the time frame."""
        return self._end_time   
    
class TimeFrameBuffer():
    """The buffer maintains :py:class:`collections.deque` of :py:class:`.TimeFrame` objects and ensures that for each such :py:class:`.TimeFrame`,
    
    * :py:attr:`TimeFrame.end_time` <= :py:attr:`TimeFrameBuffer.end_time` and
    * :py:attr:`TimeFrameBuffer.start_time` <= :py:attr:`TimeFrame.start_time`. 

    Note:
    
    * This class is intended to be used for time frames that are *contiguous*, i.e. time frames that are ascending and have no gaps in between them.  
    * The buffer is not thread-safe, i.e. it is not safe to add or remove :py:class:`.TimeFrame` objects from the buffer from multiple threads at the same time.
    
    :param max_duration: Sets the :py:attr:`~.TimeFrameBuffer.max_duration` of this buffer. Optional, default is 1.0.
    :type max_duration: float
    :param error_margin: Sets the :py:attr:`~.TimeFrameBuffer.error_margin` of this buffer. Optional, default is 1e-10.
    :type error_margin: float
    """

    def __init__(self, max_duration: float = 1.0, error_margin: float = 1e-10) -> None:
        
        # Set deque
        self._deque = deque()

        # Set max duration
        if not isinstance(max_duration, float):
            raise TypeError("The max duration must be a float.")
        if not max_duration > 0:
            raise ValueError("The max duration must be greater than 0.")
        self._max_duration = max_duration

        # Set error margin
        if not isinstance(error_margin, float):
            raise TypeError("The error margin must be a float.")
        if not error_margin >= 0:
            raise ValueError("The error margin must be at least 0.")
        self._error_margin = error_margin
    
    @property
    def max_duration(self) -> float:
        """:return: The positive float indicating the maximum duration of this buffer.
        :rtype: float"""
        return self._max_duration

    @property
    def error_margin(self) -> float:
        """:return: The non-negative float indicating the error margin of this buffer used to determine whether two consecutive :py:class:`.TimeFrame` objects in the buffer occur directly after one another.
        :rtype: float"""
        return self._error_margin

    @property
    def start_time(self) -> float:
        """:raises Exception: If the :py:meth:`time_frame_count` is 0.
        :return: The start_time of the earliest :py:class:`.TimeFrame` in the buffer.
        :rtype: float
        """
        if self.time_frame_count == 0:
            raise Exception("Cannot obtain start_time for empty TimeFrameBuffer.")
        else:
            return self._deque[0].start_time
    
    @property
    def duration(self) -> float:
        """:return: The duration of this buffer. This is the time between the :py:attr:`~.TimeFrame.start_time` of the earliest :py:class:`.TimeFrame` and the :py:attr:`~.TimeFrame.end_time` of the latest :py:class:`.TimeFrame`. Returns 0 in case the buffer is empty.
        :rtype: float"""
        if len(self._deque) == 0:
            return 0.0
        else:
            return self._deque[-1].end_time - self._deque[0].start_time

    @property
    def end_time(self) -> float:
        """:raises Exception: If the :py:meth:`time_frame_count` is 0.
        :return: The :py:attr:`~.TimeFrame.end_time` of the latest :py:class:`.TimeFrame` in the :py:class:`.TimeFrameBuffer`. 
        :rtype: float"""
        if self.time_frame_count == 0:
            raise Exception("Cannot obtain end_time for empty TimeFrameBuffer.")
        else:
            return self._deque[-1].end_time
    
    @property
    def time_frame_count(self) -> int:
        """:return: The number of :py:class:`.TimeFrame` objects currently in the buffer.
        :rtype: int"""
        return len(self._deque)

    def insert(self, time_frame: TimeFrame) -> None:
        """Appends the given **time_frame** to the latest end of the :py:attr:`.TimeFrameBuffer`. 

        :param time_frame: The :py:class:`.TimeFrame` to insert at the latest end of the buffer.
        :type time_frame: :py:class:`.TimeFrame`
        :raises ValueError: If the caller attempts to insert a :py:class:`.TimeFrame` whose :py:attr:`~.TimeFrame.duration` is longer than the :py:attr:`.TimeFrameBuffer.duration`.
        :raises ValueError: If the insertion would break contiguity, i.e. if buffer contains :py:class:`.TimeFrame` objects and the caller attempts to insert a non :py:class:`.TimeFrame` for which the absolute difference between the :py:attr:`.TimeFrame.start_time` of the provided **time_frame** and the :py:attr:`~.TimeFrame.end_time` of the latest :py:class:`.TimeFrame` in the buffer is larger than :py:attr:`.TimeFrameBuffer.error_margin`.
        """

        # Append new time frame to the buffer
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        if not time_frame.duration < self.duration:
            raise ValueError(f"Cannot insert a time frame of duration {time_frame.duration} into a buffer whose maximum duration is {self.duraiton}.")
        if not abs(time_frame.start_time - self.end_time) <= self._error_margin:
            raise ValueError("The start time of the to be inserted time frame must be equal to the current end time of the buffer, up to the accepted error margin.")
        self._deque.insert_at_head(data=time_frame)

        # Remove the oldest time frames from the buffer
        while len(self._deque) > 1 and self._deque[0].start_time < self.end_time-self.max_duration:
            self._deque.popleft()



class Port():
    """A class that maintains sends or receives :py:class:`.TimeFrame` objects for a pair of :py:class:`.Area` objects.

    :param time_frame_buffer: Sets the :py:attr:`~.Port.time_frame_buffer` property of this port.
    :type time_frame_buffer: :py:class:`.TimeFrameBuffer`
    :param time_axis: Sets the :py:attr:`~.Port.time_axis` property of this port.
    :type time_axis: int
    :param input_area: Sets the :py:attr:`~Port.input_area` of this port. Optional, defaults to None. At least one of [**input_area**, **output_area**] needs to be provided.
    :type input_area: :py:class:`.Area`, optional
    :param output_area: Sets the :py:attr:`~Port.output_area` of this port. Optional, defaults to None. At least one of [**input_area**, **output_area**] needs to be provided.
    :type output_area: :py:class:`.Area`, optional
    """

    def __init__(self, time_frame_buffer: TimeFrameBuffer, time_axis: int, connection: "Connection", input_area: 'Area'=None, output_area: 'Area'=None) -> None:

        # Set the buffer
        if not isinstance(time_frame_buffer, TimeFrameBuffer):
            raise TypeError("The time_frame_buffer must be a TimeFrameBuffer object.")
        self._time_frame_buffer = time_frame_buffer

        # Set time axis
        if not isinstance(time_axis, int):
            raise TypeError("The time_axis must be an int.")
        if not time_axis >= 0:
            raise ValueError("The time_axis must be greater than or equal to 0.")
        self._time_axis = time_axis

        # Set area references
        if input_area is None and output_area is None:
            raise ValueError("At least one of input_area or output_area must be specified.")
        
        if not isinstance(input_area, Area):
            raise TypeError("Input area must be a Area object.")
        self._input_area = input_area

        if not isinstance(output_area, Area):
            raise TypeError("Output area must be a Area object.")
        self._output_area = output_area

    @property
    def time_frame_buffer(self) -> TimeFrameBuffer:
        """:return: The :py:class:`.TimeFrameBuffer` of this port.
        :rtype: :py:class:`TimeFrameBuffer`
        """
        return self._time_frame_buffer
    
    @property
    def time_axis(self) -> int:
        """:return: The axis along which time passes across :py:class:`.TimeFrame` objects in the :py:class:`.TimeFrameBuffer` of this port. This is useful if the caller intends to concatenate the :py:class:`.TimeFrame` objects.
        :rtype: int
        """
        return self._time_axis
    
    @property
    def input_area(self) -> "Area":
        """:return: The :py:class:`.Area` from which this port receives its :py:class:`.TimeFrame` objects.
        :rtype: :py:class:`.Area`
        """
        return self._input_area
    
    @property
    def output_area(self) -> "Area":
        """:return: The :py:class:`.Area` to which this port sends its :py:class:`.TimeFrame` objects.
        :rtype: :py:class:`.Area`
        """
        return self._input_area
    
    
    def send(self, time_frame: TimeFrame) -> None:
        """Sends the given **time_frame** via the given **connection** to :py:meth:`~.Port.output_area`.

        :param time_frame: The :py:class:`.TimeFrame` to be sent.
        :type time_frame: :py:class:`.TimeFrame`
        :param connection: The :py:class:`.Connection` via which the **time_frame** shall be sent.
        :type connection: Connection
        :raises Exception: _description_
        """

        if self.output_area == None:
            raise Exception("Cannot send time frame to area None.")
    


    def insert_time_frame(self, time_frame: TimeFrame) -> None:
        """Places the new **time_frame** at the recent end of :py:attr:`~.Port.buffer` to the end of the :py:attr:`~.Port.state` and pushes the first :py:class:`.TimeFrame` out of the :py:attr:`~.Port.state`.

        :param time_frame: The time frame to insert into the state of the port.
        :type time_frame: TimeFrame
        """
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        self._time_frame_buffer.insert(time_frame=time_frame)

class Area(nn.Module):
    _count = 0
    """A counter for the number of areas. This is used to assign a unique index to each area."""

    def __init__(self, initial_state: torch.Tensor | List[torch.Tensor], input_ports: Dict[int, Port], output_port: Port, update_rate: float, transformation: nn.Module) -> None:
        """A area corresponds to a small population of biological neurons that jointly hold one representation. It has a state that is updated by transforming and aggregating inputs from other areas.
        
        :param initial_state: Sets the :py:attr:`~.Area.state` of this area. 
        :type initial_state: :py:class:`~torch.Tensor` | List[:py:class:`~torch.Tensor`]
        :param input_ports: Sets the :py:attr:`~.Area.input_ports` of this area. 
        :type input_ports: Dict[int, Port]
        :param output_port: Sets the :py:attr:`~.Area.output_port` of this area.
        :type output_port: Port
        :param update_rate: Sets the :py:attr:`~.Area.update_rate` of this area.
        :type update_rate: float
        :param transformation: Sets the :py:attr:`~.Area.transformation` of this area.
        :type transformation: torch.nn.Module
        """

        # Call the parent constructor
        super().__init__()
        
        # Indexing
        self._index = Area._count; Area._count += 1
        
        # States
        if isinstance(initial_state, list):
            if len(initial_state) == 0:
                raise ValueError("The initial state must be a non-empty list.")
            for i in range(len(initial_state)):
                if not isinstance(initial_state[i], torch.Tensor):
                    raise TypeError("The initial state must be a list of :py:class:`~torch.Tensor` objects.")
        elif not isinstance(initial_state, torch.Tensor):
            raise TypeError("Initial state must be a :py:class:`~torch.Tensor`.")
        self._state = initial_state

        # Ports
        if not isinstance(input_ports, dict):
            raise TypeError("Input ports must be a dictionary where each key is an area index and each value is a port object.")
        if not all(isinstance(port, Port) for port in input_ports.values()):
            raise TypeError("All values in the input ports dictionary must be Port objects.")
        self._input_ports = input_ports
        
        if output_port is not None:
            if not isinstance(output_port, Port):
                raise TypeError("If the output port is provided, it must be a Port object.")
        self._output_port = output_port
        
    @property
    def index(self) -> int:
        """The index of the area. This is used to identify the area in the network."""
        return self._index

    @property
    def state(self) -> torch.Tensor | List[torch.Tensor]:
        """The state of the area. This can be a :py:class:`~torch.Tensor`, for instance of shape [instance count, dimensionality] or a list of :py:class:`~torch.Tensor` objects."""
        return self._state

    @property
    def input_ports(self) -> Dict[int, Port]:
        """A dictionary that maps from an input area's index to a corresponding :py:class:`~network_components.Port` object that buffers the input. It may be an empty dict if the area is the input area of the network."""
        return self._input_ports
    
    @property
    def output_port(self) -> Port:
        """The output port of the area. This is a :py:class:`~network_components.Port` object that buffers the output of the area. It may be None if the area is the output area of the network."""
        return self._output_port

    def forward(self) -> None:
        """Advances the area by 1 time step by letting it transform the inputs from its :py:attr:`~.Area.input_ports` and pass the result to its :py:attr:`~.Area.output_port`. """
        
        if self._input_ports is not None:
            # TODO: Implement the forward pass for the area. This should involve transforming the inputs from the input ports and passing the result to the output port.
            # For now, we just sum the inputs from all input ports and set the state to that sum
            pass

        return

class Connection(nn.Module):

    def __init__(self, input_area: Area, output_area: Area, transformation: nn.Module = nn.Identity()) -> None:
        """A connection between two :py:class:`.Area` objects. This is analogous to a group of synapses in a biological neural network that process a representation between the input and the output area.

        :param input_area: Sets the :py:attr:`~.Connection.input_area` of the connection. 
        :type input_area: Area
        :param output_area: Sets the :py:attr:`~.Connection.output_area` of the connection.
        :type output_area: Area
        :param transformation: Sets the :py:attr:`~.Connection.transformation` of the connection. 
        :type transformation: nn.Module
        """

        # Call the parent constructor
        super().__init__()

        # Set area references
        if not isinstance(input_area, Area):
            raise TypeError("Input area must be a Area object.")
        self._input_area = input_area
        if not isinstance(output_area, Area):
            raise TypeError("Output area must be a Area object.")
        self._output_area = output_area

        # Set ports
        input_area.add_port(port=Port(initial_state=input_area.state, input_area=input_area, output_area=output_area))
        
        # Set transformation
        if not isinstance(transformation, nn.Module):
            raise TypeError("Transformation must be a nn.Module object.")
        self._transformation = transformation
        
    @property
    def input_area(self) -> Area:
        """The area that is the source of the connection."""
        return self._input_area
    
    @property
    def output_area(self) -> Area:
        """The area that is the target of the connection."""
        return self._output_area

    @property
    def transformation(self) -> nn.Module:
        """The transformation that is applied to the input to obtain the output."""
        return self.__transformation
    
    def forward(self) -> None:
        """Applies the :py:attr:`~.Connection.transformation` to the data of the :py:attr:`~.Connection.input_port` and uses the result to set the data of the :py:attr:`~.Connection.output_port`.
        """
        self.__output_node.input_from_area(input=self.__transformation(self.__input_node.__state), area=self.__input_node)
    
class BrIANN():
    def __init__(self, adjacency_matrix: np.ndarray):
        """"""
        pass

    def step(self) -> None:
        """Advances the network by 1 time step by letting each area transform the inputs from its :py:attr:`~.Area.input_ports` and pass the result to its :py:attr:`~.Area.output_port`. """
        for area in self._areas:
            area.forward()
        return

'''
class InputAccumulationStrategy(ABC):
    """Abstract base class for input aggregation strategies. A strategy provides a method to curate the input from the :py:attr:`~.Area.input_ports` of an :py:class:`.Area` such that it can transform its input to an output."""

    @abstractmethod
    def curate(self, input_ports: Dict[int, Port]) -> torch.Tensor:
        """Curates a tensor from the :py:attr:`~.Area.input_ports` of an :py:class:`.Area` such that it can transform its input to an output. The input ports are assumed to be in the same order as the areas in the network."""
        pass

class SummativeInputAggregationStrategy(InputAccumulationStrategy):

    def __init__(self) -> None:
        """An input aggregation strategy that sums the inputs from all input ports into a single :py:class:`~torch.Tensor`. It is assumed that the :py:attr:`~.Port.time_frame`s of all input input :py:class:`.Port` have the same shape."""
        super().__init__()
    
    def curate(self, input_ports: Dict[int, Port]) -> torch.Tensor:
        """Sums the inputs from all input :py:class:`.Port`s into a single :py:class:`~torch.Tensor`. It is assumed that the :py:attr:`~.Port.time_frame`s of all input input :py:class:`.Port` have the same shape."""
        if not isinstance(input_ports, dict):
            raise TypeError("Input ports must be a dictionary where each key is an area index and each value is a port object.")
        if not all(isinstance(port, Port) for port in input_ports.values()):
            raise TypeError("All values in the input ports dictionary must be Port objects.")
        
        # Sum the inputs from all input ports
        tmp = torch.cat([port.time_frame[torch.newaxis,:] for port in input_ports.values()], dim=0)
        return tmp.sum(dim=0)

class Package():
    """A class that stores minimal meta-data about a :py:class:`.TimeFrame` object to be sent from one :py:class:`.Area` to another.
    
    :param time_frame: The :py:class:`.TimeFrame` object to be sent.
    :type time_frame: :py:class:`.TimeFrame`
    :param sender: The :py:class:`.Area` from which the **time_frame** is sent.
    :type sender: :py:class:`.Area`
    :param receiver: The :py:class:`.Area` to which the **time_frame** is sent.
    :type sender: :py:class:`.Area`
    """

    def __init__(self, time_frame: TimeFrame, sender: "Area", receiver: "Area"):
        
        # Set properties
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        self._time_frame = time_frame

        if not isinstance(sender, Area):
            raise TypeError("The sender must be an Area object.")
        self._sender = sender

        if not isinstance(receiver, Area):
            raise TypeError("The receiver must be an Area object.")
        self._sender = sender

    @property
    def time_frame(self) -> TimeFrame:
        """The content of this package."""
        return self._time_frame
    
    @property
    def sender(self) -> "Area":
        """The :py:class:`.Area` from which the package is sent."""
        return self._sender
    
    @property
    def receiver(self) -> "Area":
        """The :py:class:`.Area` to which the package is sent."""
        return self._receiver        
        
'''