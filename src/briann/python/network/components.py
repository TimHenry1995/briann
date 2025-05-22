"THIS IS A TEST COMMEMNT"

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
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
    
    * This class is intended to be used for time frames that are *contiguous*, i.e. time frames that are ascending and have no gaps larger than :py:meth:`TimeFrameBuffer.ERROR_MARGIN` in between them.  
    * The buffer is not thread-safe, i.e. it is not safe to add or remove :py:class:`.TimeFrame` objects from the buffer from multiple threads at the same time.
    """

    ERROR_MARGIN = 1e-10 
    """The error margin used to determine whether two time frames are contiguous. This is a static constant."""

    def __init__(self) -> None:
            
        # Set deque
        self._deque = deque()
    
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
        :raises ValueError: If the insertion would break contiguity, i.e. if buffer contains :py:class:`.TimeFrame` objects and the caller attempts to insert a non :py:class:`.TimeFrame` for which the absolute difference between the :py:attr:`.TimeFrame.start_time` of the provided **time_frame** and the :py:attr:`~.TimeFrame.end_time` of the latest :py:class:`.TimeFrame` in the buffer is larger than :py:attr:`.TimeFrameBuffer.ERROR_MARGIN`.
        """

        # Append new time frame to the buffer
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        if not abs(time_frame.start_time - self.end_time) <= TimeFrameBuffer.ERROR_MARGIN:
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
            
class Connection(nn.Module):
    """A connection between an :py:class:`Area` and a :py:class:`.TimeFrameBuffer`. This is analogous to a neural tract between areas of a biological neural network that not only sends information but also converts it between the reference frames of the input and output area.

    :param input_area_index: Sets the :py:meth:`~Connection.input_area_index` of this connection. 
    :type input_area_index: int
    :param output_area_index: Sets the :py:meth:`~Connection.output_area_index` of this connection. 
    :type output_area_index: int
    :param output_time_frame_buffer: Sets the :py:meth:`~Connection.output_time_frame_buffer` of this connection.
    :type output_time_frame_buffer: :py:class:`.TimeFrameBuffer`
    :param transformation: Sets the :py:meth:`~.Connection.transformation` of the connection.
    :type transformation: torch.nn.Module, optional, defaults to :py:class:`torch.nn.Identity`
    """

    def __init__(self, input_area_index: int, output_area_index: int, output_time_frame_buffer: TimeFrameBuffer, transformation: nn.Module = nn.Identity()) -> None:
        
        # Call the parent constructor
        super().__init__()

        # Set input area index
        if not isinstance(input_area_index, int):
            raise TypeError("The input_area_index must be an int.")
        self._input_area_index = input_area_index

        # Set output area index
        if not isinstance(output_area_index, int):
            raise TypeError("The output_area_index must be an int.")
        self._output_area_index = output_area_index

        # Set output time frame buffer
        if not isinstance(output_time_frame_buffer, TimeFrameBuffer):
            raise TypeError("The output_time_frame_buffer must be a TimeFrameBuffer object.")
        self._output_time_frame_buffer = output_time_frame_buffer
        
        # Set transformation
        if not isinstance(transformation, nn.Module):
            raise TypeError("Transformation must be a nn.Module object.")
        self._transformation = transformation

    @property
    def input_area_index(self) -> int:
        """:return: The index of the area that is the source of the connection.
        :rtype: int
        """
        return self._input_area_index
    
    @property
    def output_area_index(self) -> int:
        """:return: The index of the area that is the target of the connection.
        :rtype: int
        """
        return self._output_area_index
    
    @property
    def output_time_frame_bufer(self) -> TimeFrameBuffer:
        """:return: The time frame buffer that is the target of the connection.
        :rtype: :py:class:`.TimeFrameBuffer`
        """
        return self._output_time_frame_buffer

    @property
    def transformation(self) -> nn.Module:
        """:return: The transformation that is applied to the input to obtain the output.
        :rtype: torch.nn.Module
        """
        return self.__transformation
    
    def forward(self, time_frame: TimeFrame) -> None:
        """Applies the :py:meth:`~.Connection.transformation` to the given **time_frame** and inserts it into the :py:meth:`~.Connection.output_time_frame_buffer` of self.
        """

        # Check if the time frame is valid
        if not isinstance(time_frame, TimeFrame):
            raise TypeError("The time frame must be a TimeFrame object.")
        
        # Apply the transformation to the time frame
        transformed_time_frame = self._transformation(time_frame.state)
        
        # Create a new time frame with the transformed state
        new_time_frame = TimeFrame(state=transformed_time_frame, index=time_frame.index, start_time=time_frame.start_time, duration=time_frame.duration)
        
        # Insert the new time frame into the output time_frame_buffer's buffer
        self._output_time_frame_buffer.insert(new_time_frame)
  
class Area(nn.Module):
    """A area corresponds to a small population of biological neurons that jointly hold one representation. It has a state that is updated by transforming and aggregating inputs from other areas.
        
    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param initial_state: Sets the :py:meth:`~.Area.state` of this area. 
    :type initial_state: torch.Tensor | List[torch.Tensor]
    :param input_time_frame_buffers: Sets the :py:meth:`~.Area.input_time_frame_buffers` of this area. 
    :type input_time_frame_buffers: Dict[int, :py:class:`.TimeFrameBuffer`]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`.Connection`]
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param processing_time: Sets the :py:meth:`~.Area.processing_time` of this area.
    :type processing_time: float, optional, defaults to 0
    """
    
    TIME_AXIS = 1
    """The time axis of this area. This is the axis along which the time frames are concatenated when being read from a :py:class:`.TimeFrameBuffer`."""

    def __init__(self, index: int, initial_state: torch.Tensor | List[torch.Tensor], input_time_frame_buffers: Dict[int, TimeFrameBuffer], output_connections: Dict[int, Connection], update_rate: float, transformation: nn.Module, processing_time: float = 0) -> None:
        
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

        # Set input time_frame_buffers
        if not isinstance(input_time_frame_buffers, dict):
            raise TypeError(f"The input_time_frame_buffers for area {index} must be a dictionary where each key is an area index and each value is a TimeFrameBuffer object.")
        if not all(isinstance(time_frame_buffer, TimeFrameBuffer) for time_frame_buffer in input_time_frame_buffers.values()):
            raise TypeError(f"All values in the input_time_frame_buffers dictionary of area {index} must be TimeFrameBuffer objects.")
        self._input_time_frame_buffers = input_time_frame_buffers
        
        # Output connections
        if not isinstance(output_connections, dict):
            raise TypeError(f"Output connections for area {index} must be a dictionary where each key is an area index and each value is a Connection object.")
        if not all(isinstance(connection, Connection) for connection in output_connections.values()):
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
    def input_time_frame_buffers(self) -> Dict[int, TimeFrameBuffer]:
        """:return: A dictionary where each key is an area index and each value is a TimeFrameBuffer object. These input buffers accumulate incoming :py:class:`.TimeFrame` objects.
        :rtype: Dict[int, TimeFrameBuffer]
        """
        return self._input_time_frame_buffers
    
    @property
    def output_connections(self) -> Dict[int, Connection]:
        """:return: A dictionary where each key is an area index and each value is a :py:class:`.Connection` object. These output connections are used to send the updated state of this area to the :py:class:`.TimeFrameBuffer` objects of other areas.
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
        """Processes the time frames from the :py:meth:`~.Area.input_time_frame_buffers` using :py:meth:`~.Area.transformation` of self and passes the result to the :py:meth:`~.Area.output_connections`.
        As a side-effect, the :py:meth:`~.Area.input_time_frame_buffers` of this area cleared and the :py:meth:`~.Area.state` of this area is updated.
        """

        # Iterate all input time_frame_buffers and clear them
        states = {}
        longest_duration = sys.float_info.min # The longest duration of buffers among the input_time_frame_buffers
        current_time = sys.float_info.min # The current time of the latest time frame
        for area_index, time_frame_buffer in self._input_time_frame_buffers.items():
            
            # Update the times
            longest_duration = max(longest_duration, time_frame_buffer.time_frame_buffer.duration)
            current_time = max(current_time, time_frame_buffer.time_frame_buffer.end_time)

            # Get all time frames from the current input_time_frame_buffer
            time_frames = time_frame_buffer.time_frame_buffer.clear()
            if len(time_frames) > 0:
                states[area_index] = torch.concat([time_frame.state for time_frame in time_frames], dim=Area.TIME_AXIS)
                
            else:
                states[area_index] = None

        # Apply transformation to the states
        self._state = self._transformation.forward([self._state, states])

        # Create a new time frame for the current state
        new_time_frame = TimeFrame(state=self._state, index=self._produced_time_frames, start_time=current_time+self._processing_time-longest_duration, duration=longest_duration)
        self._produced_time_frames += 1

        # Pass the transformed states to the output connections
        for area_index, connection in self._output_connections.items():
            connection.forward(time_frame=new_time_frame)

        return

class TimeAverageThenStateConcatenateThenTransformLinear(nn.Module):
    """An :py:class:`.Area` that first averages the time frames for each input buffer, then concatenates them together with the area's state along their last axis and finally applies a linear transformation.
    Note:
    
    * The state of the calling :py:class:`.Area` is assumed to be a mere torch.Tensor object, i.e. it is not a list of torch.Tensor objects.
    * The input states of the other :py:class:`.Area`s are assumed to be torch.Tensor objects of shape [instance count, time frame count, dimensionality]. 

    :param input_dimensionality: The dimensionality of the input. This is the sum of the dimensionalities of all inputs (inclduing the state of the calling area).
    :type input_dimensionality: int
    :param output_dimensionality: The dimensionality that the output state shall have.
    :type output_dimensionality: int
    """

    def __init__(self, input_dimensionality: int, output_dimensionality: int) -> None:
        
        # Call the parent constructor
        super().__init__()

        # Set linear transformation
        self.linear = nn.Linear(input_dimensionality, output_dimensionality)

    def forward(self, inputs: Tuple[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Applies the main calculation of this module to the given **inputs** and returns the result.

        :param inputs: A tuple of two elements. The first element corresponds to the :py:meth:`~.Area.state` of the calling area. The second element is a list of torch.Tensor objects that are extracted and concatenated from each of the :py:meth:`~.Area.input_time_frame_buffers` of the calling area. 
        :type inputs: Tuple[torch.Tensor, List[torch.Tensor]]
        :return: The result of the main calculation.
        :rtype: torch.Tensor
        """
        # Unpack inputs
        calling_area_state, input_states = inputs

        # Concatenate averaged the time frames
        concatenated = [None] * len(input_states) + 1
        concatenated[0] = calling_area_state
        for i, input_state in enumerate(input_states):
            concatenated[i+1] = torch.mean(input_state, dim=Area.TIME_AXIS)
        concatenated = torch.concat(concatenated, dim=-1)
        
        # Apply linear transformation
        return self.linear(concatenated)

# TODO: Use a main loop and then based on update rates of each area, call the forward method of each area
class BrIANN():
    def __init__(self, adjacency_matrix: np.ndarray):
        """"""
        pass
