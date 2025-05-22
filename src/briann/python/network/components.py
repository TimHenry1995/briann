"THIS IS A TEST COMMEMNT"

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
import sys
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
    
    :param time_axis: Sets the :py:attr:`~.TimeFrameBuffer.time_axis` property of this buffer.
    :type time_axis: int
    :param error_margin: Sets the :py:attr:`~.TimeFrameBuffer.error_margin` of this buffer.
    :type error_margin: float, optional, defaults to 1e-10
    """

    def __init__(self, time_axis: int, error_margin: float = 1e-10) -> None:
                
        # Set time axis
        if not isinstance(time_axis, int):
            raise TypeError("The time_axis must be an int.")
        if not time_axis >= 0:
            raise ValueError("The time_axis must be greater than or equal to 0.")
        self._time_axis = time_axis

        # Set error margin
        if not isinstance(error_margin, float):
            raise TypeError("The error margin must be a float.")
        if not error_margin >= 0:
            raise ValueError("The error margin must be at least 0.")
        self._error_margin = error_margin

        # Set deque
        self._deque = deque()
    
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
        :raises ValueError: If the insertion would break contiguity, i.e. if buffer contains :py:class:`.TimeFrame` objects and the caller attempts to insert a non :py:class:`.TimeFrame` for which the absolute difference between the :py:attr:`.TimeFrame.start_time` of the provided **time_frame** and the :py:attr:`~.TimeFrame.end_time` of the latest :py:class:`.TimeFrame` in the buffer is larger than :py:attr:`.TimeFrameBuffer.error_margin`.
        """

        # Append new time frame to the buffer
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        if not abs(time_frame.start_time - self.end_time) <= self._error_margin:
            raise ValueError("The start time of the to be inserted time frame must be equal to the current end time of the buffer, up to the accepted error margin.")
        self._deque.insert_at_head(data=time_frame)

    def pop(self) -> TimeFrame:
        """Removes the :py:class:`.TimeFrame` at the latest end of the buffer and returns it.

        :return: The :py:class:`.TimeFrame` at the latest end of the buffer.
        :rtype: :py:class:`.TimeFrame
        :raises Exception: If the buffer is empty.
        """
        if self.time_frame_count == 0:
            raise Exception("Cannot pop from empty TimeFrameBuffer.")
        else:
            return self._deque.pop()
    
    def clear(self) -> List[TimeFrame]:
        """Removes all :py:class:`.TimeFrame` objects from the buffer and returns them.
        :return: A list of all :py:class:`.TimeFrame` objects that were in the buffer.
        :rtype: List[:py:class:`.TimeFrame`]
        """
        if self.time_frame_count == 0:
            return []
        else:
            tmp = list(self._deque)
            self._deque.clear()
            return tmp
    
class Port():
    """A port is supposed to sit at an area for which it links a :py:class:`.TimeFrameBuffer` to a :py:class:`.Connection` such that time frames can be sent or received.
    Every :py:class:`.Area` should have at least one :py:class:`.Port` as input and at least one as output. An exception to this are the source and sink nodes of the overall model which only need to have input ports OR output ports.
    
    :param area: Sets the :py:attr:`~.Port.area` property of this port.
    :type area: :py:class:`.Area`
    :param time_frame_buffer: Sets the :py:attr:`~.Port.time_frame_buffer` property of this port.
    :type time_frame_buffer: :py:class:`.TimeFrameBuffer`
    :param connection: Sets the :py:attr:`~.Port.connection` property of this port.
    :type connection: :py:class:`.Connection`
    """

    def __init__(self, area: "Area", time_frame_buffer: TimeFrameBuffer, connection: "Connection" = None) -> None:

        # Set area
        if not isinstance(area, Area):
            raise TypeError("The area must be a Area object.")
        self._area = area

        # Set the buffer
        if not isinstance(time_frame_buffer, TimeFrameBuffer):
            raise TypeError("The time_frame_buffer must be a TimeFrameBuffer object.")
        self._time_frame_buffer = time_frame_buffer

        # Set connection
        if not isinstance(connection, Connection):
            raise TypeError("The connection must be a Connection object.")
        self._connection = connection

    @property
    def area(self) -> "Area":
        """:return: The :py:class:`.Area` to which this port belongs.
        :rtype: :py:class:`.Area`
        """
        return self._area

    @property
    def time_frame_buffer(self) -> TimeFrameBuffer:
        """:return: The :py:class:`.TimeFrameBuffer` of this port.
        :rtype: :py:class:`TimeFrameBuffer`
        """
        return self._time_frame_buffer
    
    def send(self) -> None:
        """Passes all :py:class:`.TimeFrame` objects that are currently in the buffer to the :py:meth:`~.Port.connection` of self.
        """

        # Send all time frames in the buffer to the connection
        while self.time_frame_buffer.time_frame_count > 0:
            self._connection.forward(time_frame=self._time_frame_buffer.pop())
                    
class Connection(nn.Module):
    """A connection between two :py:class:`.Port` objects. This is analogous to a neural tract between areas of a biological neural network that not only sends information but also converts it between the reference frames of the input and output area.

    :param input_area: Sets the :py:meth:`~Connection.input_area` of this connection. 
    :type input_area: :py:class:`.Area`
    :param output_port: Sets the :py:meth:`~Connection.output_port` of this connection.
    :type output_port: :py:class:`.Port`
    :param transformation: Sets the :py:meth:`~.Connection.transformation` of the connection.
    :type transformation: torch.nn.Module, optional, defaults to :py:class:`torch.nn.Identity`
    """

    def __init__(self, input_area: "Area", output_port: Port, transformation: nn.Module = nn.Identity()) -> None:
        
        # Call the parent constructor
        super().__init__()

        # Set input area
        if not isinstance(input_area, Area):
            raise TypeError("Input area must be an Area object.")
        self._input_area = input_area

        # Set output port
        if not isinstance(output_port, Port):
            raise TypeError("Output port must be a Port object.")
        self._output_port = output_port
        
        # Set transformation
        if not isinstance(transformation, nn.Module):
            raise TypeError("Transformation must be a nn.Module object.")
        self._transformation = transformation

    @property
    def input_area(self) -> "Area":
        """:return: The area that is the source of the connection.
        :rtype: :py:class:`.Area`
        """
        return self._input_area
    
    @property
    def output_port(self) -> Port:
        """:return: The port that is the target of the connection.
        :rtype: :py:class:`.Port`
        """
        return self._output_port

    @property
    def transformation(self) -> nn.Module:
        """:return: The transformation that is applied to the input to obtain the output.
        :rtype: torch.nn.Module
        """
        return self.__transformation
    
    def forward(self, time_frame: TimeFrame) -> None:
        """Applies the :py:meth:`~.Connection.transformation` to the given **time_frame** and inserts it into the :py:meth:`~.Port.time_frame_buffer` of the :py:meth:`~.Connection.output_port` of self.
        """

        # Check if the time frame is valid
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        
        # Apply the transformation to the time frame
        transformed_time_frame = self._transformation(time_frame.state)
        
        # Create a new time frame with the transformed state
        new_time_frame = TimeFrame(state=transformed_time_frame, index=time_frame.index, start_time=time_frame.start_time, duration=time_frame.duration)
        
        # Insert the new time frame into the output port's buffer
        self._output_port.time_frame_buffer.insert(new_time_frame)
  
class Area(nn.Module):
    """A area corresponds to a small population of biological neurons that jointly hold one representation. It has a state that is updated by transforming and aggregating inputs from other areas.
        
    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param initial_state: Sets the :py:meth:`~.Area.state` of this area. 
    :type initial_state: torch.Tensor | List[torch.Tensor]
    :param input_ports: Sets the :py:meth:`~.Area.input_ports` of this area. 
    :type input_ports: Dict[int, :py:class:`.Port`]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`.Connection`]
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param processing_time: Sets the :py:meth:`~.Area.processing_time` of this area.
    :type processing_time: float, optional, defaults to 0
    """
    
    def __init__(self, index: int, initial_state: torch.Tensor | List[torch.Tensor], input_ports: Dict[int, Port], output_connections: Dict[int, Connection], update_rate: float, transformation: nn.Module, processing_time: float = 0) -> None:
        
        # Call the parent constructor
        super().__init__()
        
        # Set index
        if not isinstance(index, int):
            raise TypeError("The index must be an int.")
        self._index = index
        
        # Set state
        if isinstance(initial_state, list):
            if len(initial_state) == 0:
                raise ValueError(f"The initial state of area {index} must not be an empty list.")
            for i in range(len(initial_state)):
                if not isinstance(initial_state[i], torch.Tensor):
                    raise TypeError(f"If the initial state of area {index} is a list, it must be a list of :py:class:`~torch.Tensor` objects.")
        elif not isinstance(initial_state, torch.Tensor):
            raise TypeError(f"Initial state of area {index} must be a :py:class:`~torch.Tensor`.")
        self._state = initial_state

        # Input ports
        if not isinstance(input_ports, dict):
            raise TypeError(f"Input ports for area {index} must be a dictionary where each key is an area index and each value is a port object.")
        if not all(isinstance(port, Port) for port in input_ports.values()):
            raise TypeError(f"All values in the input_ports dictionary of area {index} must be Port objects.")
        self._input_ports = input_ports
        
        # Output connections
        if not isinstance(output_connections, dict):
            raise TypeError(f"Output connections for area {index} must be a dictionary where each key is an area index and each value is a Connection object.")
        if not all(isinstance(port, Connection) for port in output_connections.values()):
            raise TypeError(f"All values in the output_connections dictionary of area {index} must be Connection objects.")
        self._output_connections = output_connections

        # Set update rate
        if not isinstance(update_rate, float):
            raise TypeError(f"The update rate of area {index} must be a float.")
        if not update_rate > 0:
            raise ValueError(f"The update rate of area {index} must be greater than 0.")    
        self._update_rate = update_rate

        # Set transformation
        if not isinstance(transformation, nn.Module):
            raise TypeError(f"The transformation of area {index} must be a nn.Module object.")
        self._transformation = transformation

        # Set processing time
        if not isinstance(processing_time, float):
            raise TypeError(f"The processing time of area {index} must be a float.")    
        if not processing_time >= 0:
            raise ValueError(f"The processing time of area {index} must be greater than or equal to 0.")
        self._processing_time = processing_time

        # Set the number of produced time frames
        self._produced_time_frames = 0
        
    @property
    def index(self) -> int:
        """:return: The index of the area used to identify it in the overall model.
        :rtype: int"""
        return self._index

    @property
    def state(self) -> torch.Tensor | List[torch.Tensor]:
        """:return: The state of this area.
        :rtype: torch.Tensor | List[torch.Tensor]"""
        return self._state

    @property
    def input_ports(self) -> Dict[int, Port]:
        """:return: A dictionary where each key is an area index and each value is a port object. These input ports accumulate incoming :py:class:`.TimeFrame` objects.
        :rtype: Dict[int, Port]
        """
        return self._input_ports
    
    @property
    def output_connections(self) -> Dict[int, Connection]:
        """:return: A dictionary where each key is an area index and each value is a :py:class:`.Connection` object. These output connections are used to send the updated state of this area to the :py:class:`.Port` objects of other areas.
        :rtype: Dict[int, Connection]
        """
        return self._output_connections

    @property
    def update_rate(self) -> float:
        """:return: The update rate of this area.
        :rtype: float"""
        return self._update_rate
    
    @property
    def transformation(self) -> nn.Module:
        """:return: The transformation of this area.
        :rtype: torch.nn.Module"""
        return self._transformation

    @property
    def processing_time(self) -> float:
        """:return: The processing time of this area. This is a non-negative float that indicates how long it takes for this area to process the input time frames and produce an output time frame. It is not the actual processing time on the executing device but a hypothetical processing time that corresponds to the time that the corresponding brain area would take.
        :rtype: float"""
        return self._processing_time

    def forward(self) -> None:
        """Processes the time frames from the :py:meth:`~.Area.input_ports` using :py:meth:`~.Area.transformation` of self and passes the result to the :py:meth:`~.Area.output_connections`.
        As a side-effect, the :py:meth:`~.Area.input_ports` of this area cleared and the :py:meth:`~.Area.state` of this area is updated.
        """

        # Iterate all input ports and clear their buffers
        states = {self._index: self._state}
        longest_duration = sys.float_info.min # The longest duration of buffers among the input ports
        current_time = sys.float_info.min # The current time of the latest time frame
        for area_index, port in self._input_ports.items():
            
            # Update the times
            longest_duration = max(longest_duration, port.time_frame_buffer.duration)
            current_time = max(current_time, port.time_frame_buffer.end_time)

            # Get all time frames from the current input port
            time_frames = port.time_frame_buffer.clear()
            if len(time_frames) > 0:
                states[area_index] = torch.concat([time_frame.state for time_frame in time_frames], dim=port.time_frame_buffer.time_axis)
                
            else:
                states[area_index] = None

        # Apply transformation to the states
        self._state = self._transformation.forward(states)

        # Create a new time frame for the current state
        new_time_frame = TimeFrame(state=self._state, index=self._produced_time_frames, start_time=current_time+self._processing_time-longest_duration, duration=longest_duration)
        self._produced_time_frames += 1

        # Pass the transformed states to the output connections
        for area_index, connection in self._output_connections.items():
            connection.forward(time_frame=new_time_frame)

        return

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