"This module collects all necessary components to build a BrIANN model."

import numpy as np
import torch
from typing import List, Dict, Tuple, Deque
import sys
from abc import ABC, abstractmethod
from collections import deque
import json
import copy
import networkx as nx
import matplotlib.pyplot as plt

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
    
    def __init__(self, state: torch.Tensor | List[torch.Tensor], index: int, start_time: float, duration: float) -> "TimeFrame":
        
        # Set state
        if isinstance(state, list):
            if len(state) == 0:
                raise ValueError(f"The state must not be an empty list.")
            for i in range(len(state)):
                if not isinstance(state[i], torch.Tensor):
                    raise TypeError(f"If the state is a list, it must be a list of torch.Tensor objects.")
        elif not isinstance(state, torch.Tensor):
            raise TypeError(f"State must be a torch.Tensor.")
        self._state = state
        
        # Set index
        if not isinstance(index, int):
            raise TypeError("The index must be an int.")
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
        """:return: The state of the time frame. This is a :py:class:`torch.tensor` without an explicit time axis, for instance of shape [instance count, dimensionality].
        :rtype: torch.Tensor"""
        return self._state

    @property
    def index(self) -> int:
        """:return: The index of the time frame. This is used to identify a timeframe.
        :rtype: int"""
        return self._index
    
    @property
    def start_time(self) -> float:
        """:return: The start time of the time frame.
        :rtype: float"""
        return self._start_time
    
    @property
    def duration(self) -> float:
        """:return: The duration of the time frame.
        :rtype: float"""
        return self._duration

    @property
    def end_time(self) -> float:
        """:return: The end time of the time frame.
        :rtype: float"""
        return self._end_time   

    def __repr__(self) -> str:
        return f"TimeFrame(index={self._index}, start_time={self._start_time}, duration={self._duration}, end_time={self._end_time})"
    
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

    def __init__(self) -> "TimeFrameBuffer":
            
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
        if len(self._deque) > 0 and not abs(time_frame.start_time - self.end_time) <= TimeFrameBuffer.ERROR_MARGIN:
            raise ValueError("The start time of the to be inserted time frame must be equal to the current end time of the buffer, up to the accepted error margin.")
        self._deque.append(time_frame)

    def pop(self) -> TimeFrame:
        """Removes the :py:class:`.TimeFrame` at the latest end of the buffer and returns it.

        :return: The :py:class:`.TimeFrame` at the latest end of the buffer.
        :rtype: :py:class:`.TimeFrame`
        :raises Exception: If the buffer is empty.
        """
        if self.time_frame_count == 0:
            raise Exception("Cannot pop from empty TimeFrameBuffer.")
        else:
            return self._deque.pop()
    
    def clear(self, keep_latest: bool = True) -> List[TimeFrame]:
        """Removes all :py:class:`.TimeFrame` objects from the buffer and returns them.

        :param keep_latest: A flag indicating whether the most recent time frame should be kept in the buffer.
        :type keep_latest: bool, optional, defaults to True
        :return: A list of all :py:class:`.TimeFrame` objects that were in the buffer.
        :rtype: List[:py:class:`.TimeFrame`]
        """
        if self.time_frame_count == 0:
            return []
        else:
            tmp = list(self._deque)
            self._deque.clear()
            if keep_latest: self._deque.append(tmp[0])
            return tmp
        
    def __repr__(self) -> str:
        return f"TimeFrameBuffer(start_time={self.start_time}, duration={self.duration}, end_time={self.end_time}), _deque={self._deque}."
            
class Connection(torch.nn.Module):
    """A connection between an :py:class:`Area` and a :py:class:`.TimeFrameBuffer`. This is analogous to a neural tract between areas of a biological neural network that not only sends information but also converts it between the reference frames of the input and output area.
    
    :param from_area_index: Sets the :py:meth:`~Connection.from_area_index` of this connection. 
    :type from_area_index: int
    :param to_area_index: Sets the :py:meth:`~Connection.to_area_index` of this connection. 
    :type to_area_index: int
    :param intial_time_frame: Sets the :py:meth:`~Connection.initial_time_frame` of this connection.
    :type initial_time_frame: :py:class:`.TimeFrame`
    :param transformation: Sets the :py:meth:`~.Connection.transformation` of the connection.
    :type transformation: torch.nn.Module, optional, defaults to :py:class:`torch.nn.Identity`
    """

    def __init__(self, from_area_index: int, to_area_index: int, initial_time_frame: TimeFrame, transformation: torch.nn.Module = torch.nn.Identity()) -> "Connection":
        
        # Call the parent constructor
        super().__init__()

        # Set input area index
        if not isinstance(from_area_index, int):
            raise TypeError("The from_area_index must be an int.")
        self._from_area_index = from_area_index

        # Set output area index
        if not isinstance(to_area_index, int):
            raise TypeError("The to_area_index must be an int.")
        self._to_area_index = to_area_index

        # Set output time frame buffer
        self._output_time_frame_buffer = TimeFrameBuffer()
        
        # Set the initial time frame
        if not isinstance(initial_time_frame, TimeFrame):
            raise TypeError("The initial_time_frame must be a TimeFrame object.")
        self._initial_time_frame = initial_time_frame

        # Set transformation
        if not isinstance(transformation, torch.nn.Module):
            raise TypeError("Transformation must be a torch.nn.Module object.")
        self._transformation = transformation

        # Propagate the initial time frame through the transformation
        time_frame = TimeFrame(state=self.transformation(initial_time_frame.state), index=initial_time_frame.index, start_time=initial_time_frame.start_time, duration=initial_time_frame.duration)
        self._output_time_frame_buffer.insert(time_frame=time_frame)

    @property
    def from_area_index(self) -> int:
        """:return: The index of the area that is the source of the connection.
        :rtype: int
        """
        return self._from_area_index
    
    @property
    def to_area_index(self) -> int:
        """:return: The index of the area that is the target of the connection.
        :rtype: int
        """
        return self._to_area_index
    
    @property
    def output_time_frame_buffer(self) -> TimeFrameBuffer:
        """:return: The time frame buffer that is the target of the connection.
        :rtype: :py:class:`.TimeFrameBuffer`
        """
        return self._output_time_frame_buffer

    @property
    def initial_time_frame(self) -> TimeFrame:
        """:return: The time frame passed through :py:meth:`.~Connection.transformation` and then saved into the :py:meth:`~.Connection.time_frame_buffer` of self during initialization and :py:meth:`~.Connection.reset`.
        :rtype: :py:class:`.TimeFrame`
        """
        return self._initial_time_frame
    
    @property
    def transformation(self) -> torch.nn.Module:
        """:return: The transformation that is applied to the input to obtain the output.
        :rtype: torch.nn.Module
        """
        return self._transformation
        
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
        print(f"Sent time frame {time_frame} from area {self.from_area_index} to {self.to_area_index}.")
    
        # Insert the new time frame into the output time_frame_buffer's buffer
        self._output_time_frame_buffer.insert(time_frame=new_time_frame)

        
    def reset(self) -> None:
        """Resets the :py:meth:`~.Connection.output_time_frame_buffer` of this connection such that it only contains the transformed :py:meth:`.Connection.initial_time_frame`. This method should be called every time a new trial is simulated."""

        # Clear the output time frame buffer
        self.output_time_frame_buffer.clear(keep_latest=False)
        
        # Propagate the initial time frame through the transformation
        time_frame = TimeFrame(state=self.transformation(self.initial_time_frame.state), index=-1, start_time=-1.0)
        self._output_time_frame_buffer.insert(time_frame=time_frame)

    def __repr__(self) -> str:
        """Returns a string representation of the connection."""
        return f"Connection(from_area_index={self._from_area_index}, to_area_index={self._to_area_index}, transformation={self._transformation.__class__.__name__}, time_frame_buffer=\n\t{self._output_time_frame_buffer})"
  
class Area(torch.nn.Module):
    """A area corresponds to a small population of biological neurons that jointly hold one representation. It has a state that is updated by transforming and aggregating inputs from other areas.
    The area assumes that instances are enumerated along :py:attr:`~.Area.BATCH_AXIS` and that time frames are concatenated along :py:attr:`~.Area.TIME_AXIS`.    
    
    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param initial_state: Sets the :py:meth:`~.Area.state` of this area. 
    :type initial_state: torch.Tensor | List[torch.Tensor]
    :param input_time_frame_buffers: Sets the :py:meth:`~.Area.input_time_frame_buffers` of this area. 
    :type input_time_frame_buffers: Dict[int, :py:class:`.TimeFrameBuffer`]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`.Connection`]
    :param transformation: Sets the :py:meth:`~.Area.transformation` of this area.
    :type transformation: torch.nn.Module
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    """

    BATCH_AXIS = 0
    """The axis along which instances are enumerated within a batch."""
    
    TIME_AXIS = 1
    """The time axis of this area. This is the axis along which the time frames are concatenated when being read from a :py:class:`.TimeFrameBuffer`."""

    def __init__(self, index: int, initial_state: torch.Tensor | List[torch.Tensor], input_time_frame_buffers: Dict[int, TimeFrameBuffer], output_connections: Dict[int, Connection], transformation: torch.nn.Module, update_rate: float) -> "Area":
        
        # Call the parent constructor
        super().__init__()
        
        # Set index
        if not (isinstance(index, int) or isinstance(index, str)):
            raise TypeError("The index must be an int.")
        self._index = index
        
        # Set state
        if isinstance(initial_state, list):
            if len(initial_state) == 0:
                raise ValueError(f"The initial state of area {index} must not be an empty list.")
            for i in range(len(initial_state)):
                if not isinstance(initial_state[i], torch.Tensor):
                    raise TypeError(f"If the initial state of area {index} is a list, it must be a list of torch.Tensor objects.")
        elif not isinstance(initial_state, torch.Tensor):
            raise TypeError(f"Initial state of area {index} must be a torch.Tensor.")
        self._state = initial_state
        self._initial_state = initial_state

        # Set input time_frame_buffers
        if not isinstance(input_time_frame_buffers, dict):
            raise TypeError(f"The input_time_frame_buffers for area {index} must be a dictionary where each key is the index of an area from which the buffer receives input and each value is a TimeFrameBuffer object. ")
        if not all(isinstance(area_index, int) for area_index in input_time_frame_buffers.keys()):
            raise TypeError(f"All keys in the input_time_frame_buffers dictionary of area {index} must be integers.")
        if not all(isinstance(time_frame_buffer, TimeFrameBuffer) for time_frame_buffer in input_time_frame_buffers.values()):
            raise TypeError(f"All values in the input_time_frame_buffers dictionary of area {index} must be TimeFrameBuffer objects.")
        self._input_time_frame_buffers = input_time_frame_buffers
        
        # Output connections
        if not isinstance(output_connections, dict):
            raise TypeError(f"Output connections for area {index} must be a dictionary where each key is the index of the area to which the connection projects and each value is a :py:class:`.Connection` object.")
        if not all(isinstance(area_index, int) for area_index in output_connections.keys()):
            raise TypeError(f"All keys in the output_connections dictionary of area {index} must be integers.")
        if not all(isinstance(connection, Connection) for connection in output_connections.values()):
            raise TypeError(f"All values in the output_connections dictionary of area {index} must be Connection objects.")
        self._output_connections = output_connections
            
        # Set transformation
        if not isinstance(transformation, torch.nn.Module):
            raise TypeError(f"The transformation of area {index} must be a torch.nn.Module object.")
        self._transformation = transformation

        # Set update rate
        if not isinstance(update_rate, float):
            raise TypeError(f"The update rate of area {index} must be a float.")
        if not update_rate > 0:
            raise ValueError(f"The update rate of area {index} must be greater than 0.") 
        self._update_rate = update_rate

        # Set processing time
        self._processing_time = 1.0/update_rate

        # Set the number of produced time frames
        self._produced_time_frame_count = 0
        
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
        """:return: A dictionary where each key is the index of an area from which the buffer receives input and each value is a TimeFrameBuffer object. These input buffers accumulate incoming :py:class:`.TimeFrame` objects.
        :rtype: Dict[int, TimeFrameBuffer]
        """
        return self._input_time_frame_buffers
    
    @property
    def output_connections(self) -> Dict[int, Connection]:
        """:return: A dictionary where each key is the index of the area to which the connection projects and each value is a :py:class:`.Connection` object. These output connections are used to send the updated state of this area to the :py:class:`.TimeFrameBuffer` objects of other areas.
        :rtype: Dict[int, Connection]
        """
        return self._output_connections

    @property
    def transformation(self) -> torch.nn.Module:
        """:return: The transformation of this area.
        :rtype: torch.nn.Module"""
        return self._transformation
    
    @property
    def update_rate(self) -> float:
        """:return: The update rate of this area.
        :rtype: float"""
        return self._update_rate
    
    @property
    def processing_time(self) -> float:
        """:return: The processing time of this area. This is a positive float that indicates how long it takes for this area to process the input time frames and produce an output time frame. It is equal to 1/:py:meth:`~.Area.update_rate`.
        :rtype: float"""
        return self._processing_time

    @property
    def produced_time_frame_count(self) -> int:
        """The number of time frames that have been produced by this area in the current trial. This is used to index the time frames that are produced by this area.
        
        :return: The number of produced time frames.
        :rtype: int
        """
        return self._produced_time_frame_count

    def forward(self) -> None:
        """Processes the time frames from the :py:meth:`~.Area.input_time_frame_buffers` using :py:meth:`~.Area.transformation` of self and passes the result to the :py:meth:`~.Area.output_connections`.
        As a side-effect, the :py:meth:`~.Area.input_time_frame_buffers` of this area cleared and the :py:meth:`~.Area.state` of this area is updated.
        """

        # Iterate all input time_frame_buffers and clear them
        states = []
        for area_index, time_frame_buffer in self._input_time_frame_buffers.items():
            print(f"Area {self.index} has {time_frame_buffer.time_frame_count} time frames from area {area_index} in buffer.")
            
            # Get all time frames from the current input_time_frame_buffer
            time_frames = time_frame_buffer.clear(keep_latest=True)
            if len(time_frames) > 1: time_frames = time_frames[1:] # Trim off the oldest time frame since it has been processed already in the previous iteration. Note, if it is the only time-frame, it shall be processed again to smooth out the processing of the overall network
            states.append(torch.concat([time_frame.state.unsqueeze(Area.TIME_AXIS) for time_frame in time_frames], dim=Area.TIME_AXIS))
            
        # Apply transformation to the states
        self._state = self._transformation.forward([self._state, states])

        # Create a new time frame for the current state
        new_time_frame = TimeFrame(state=self._state, index=self._produced_time_frame_count, start_time=self._produced_time_frame_count*self._processing_time, duration=self._processing_time)
        self._produced_time_frame_count += 1

        # Pass the transformed states to the output connections
        for area_index, connection in self._output_connections.items():
            connection.forward(time_frame=new_time_frame)

    def reset(self) -> None:
        """Resets the :py:meth:`~.Area.state` and :py:meth:`~.Area.produced_time_frame_count` of this area to their initial values. This should be done before simulating a new trial."""
        
        # Reset the state
        self._state = self._initial_state
        
        # Reset the produced time frame count
        self._produced_time_frame_count = 0
    
    def __repr__(self) -> str:
        """Returns a string representation of the area."""
        return f"Area(index={self._index}, update_rate={self._update_rate}, produced_time_frame_count={self.produced_time_frame_count}, state shape(s)={self._state.shape if isinstance(self._state, torch.Tensor) else [s.shape for s in self._state]})"

class Source(Area):
    """The source :py:class:`.Area` is a special area because it streams the input to the other areas. In order to set it up for the simulation of a trial,
    load stimuli via the :py:meth:`~.Source.load_stimuli method. Then, during each call to the :py:meth:`.~Area.forward` method, one :py:class:`.TimeFrame` 
    will be taken from the stimuli and streamed to the other areas. Once the time frames are all streamed, the source area will use the **hold_function**
    to hold the last time frame for a specified **cool_down_duration**, to let the other areas finish their processing.

    :param index: Sets the :py:attr:`~.Area.index` of this area.
    :type index: int
    :param initial_state: Sets the :py:meth:`~.Area.state` of this area. 
    :type initial_state: torch.Tensor | List[torch.Tensor]
    :param output_connections: Sets the :py:meth:`~.Area.output_connections` of this area.
    :type output_connections: Dict[int, :py:class:`.Connection`]
    :param update_rate: Sets the :py:meth:`~.Area.update_rate` of this area.
    :type update_rate: float
    :param cool_down_duration: Sets the :py:meth`.~.Source.cool_down_duration` of the source area.
    :type cool_down_duration: float
    :param hold_function: Sets the :py:meth:`~.Source.hold_function` of the source area.
    :type hold_function: callable, optional, default to lambda last_state: torch.zeros_like(last_state)
    """

    def __init__(self, index: int, initial_state: torch.Tensor, output_connections: Dict[int, Connection], update_rate: float, cool_down_duration: float, hold_function: callable = lambda last_state: torch.zeros_like(last_state)) -> "Source":

        # Call the parent constructor
        super().__init__(index=index,
                         initial_state=initial_state,
                         input_time_frame_buffers={},
                         output_connections=output_connections,
                         transformation=torch.nn.Identity(),
                         update_rate=update_rate)
        
        # Set cool down duration
        if not isinstance(cool_down_duration, float):
            raise TypeError("The cool_down_duration must be a float.")
        if not cool_down_duration >= 0:
            raise ValueError("The cool_down_duration must be greater than or equal to 0.")
        self._cool_down_duration = cool_down_duration
        self._remaining_cool_down_duration = cool_down_duration

        # Set the hold function
        if not callable(hold_function):
            raise TypeError("The hold_function must be a callable that takes as input a tensor which is equal to the state of the last TimeFrame object generated from the input to the simulation.")
        self._hold_function = hold_function

        # Set the stimuli
        self._stimuli = deque()

    @property
    def cool_down_duration(self) -> float:
        """The cool down duration of the source area. This is the duration in seconds for which the source area shall hold the last state using :py:meth:`~.Source.hold_function` at the end of the simulation to let the other areas finish their processing.
        
        :return: The cool down duration.
        :rtype: float
        """
        return self._cool_down_duration
    
    @property
    def remaining_cool_down_duration(self) -> float:
        """The remaining :py:meth:`~.Source.cool_down_duration` to be discounted once the current stimuli have been streamed. This is used to determine the end of the simulation.

        :return: The cool down duration.
        :rtype: float
        """
        return self._cool_down_duration
    
    @property
    def hold_function(self) -> callable:
        """The hold function that is called whenever the source area has no more time frames to process. It is used to hold the state of the source areas last :py:class:`.TimeFrame` object for :py:meth:`~.Source.cool_down_duration` seconds while the remaining model areas are still processing the input.
        
        :return: The hold function.
        :rtype: callable
        """
        return self._hold_function

    def load_stimuli(self, stimuli: Deque[TimeFrame]) -> None:
        """Loads the stimuli that will be streamed to the other model areas during the simulation.

        :param stimuli: The :py:class:`.TimeFrame` objects to be be processed by the model. The time frames have to be ordered within the deque such that they will get extacted chronologically by the deque.pop method. The :py:meth:`~.TimeFrame.state` of each time frame is required to enumerate instances along the :py:attr:`.Area.BATCH_AXIS`. **IMPORTANT**: The duration of each time frame must be equal to the :py:meth:`~.Area.processing_time` of the source area, in seconds. 
        :type stimuli: Deque[:py:class:`.TimeFrame`]
        :raises ValueError: if the duration of any time frame is not equal to the :py:meth:`~.Area.processing_time` of the source area, in seconds.
        """
            
        # Check input validity
        if not isinstance(stimuli, deque):
            raise TypeError("The stimuli must be a deque of TimeFrame objects.")
        if len(stimuli) == 0:
            raise ValueError("The stimuli must not be an empty deque.")
        for time_frame in stimuli:
            if not isinstance(time_frame, TimeFrame):
                raise TypeError("Each time frame must be a TimeFrame object.")
            if not abs(time_frame.duration - self._processing_time) < 1e-10: # Accounting for small numerical errors
                raise ValueError("The duration of each time frame must be equal to the processing time of the source area, in seconds.")
    
        # Set stimuli
        self._stimuli = stimuli

        # Reset the remaining cool down duration
        self._remaining_cool_down_duration = self._cool_down_duration

    @property
    def stimuli(self) -> Deque[TimeFrame]:
        """The stimuli that are currently loaded in the source area. This is a deque of :py:class:`.TimeFrame` objects that are to be processed by the model.
        
        :return: The stimuli.
        :rtype: Deque[:py:class:`.TimeFrame`]
        """
        return self._stimuli

    def forward(self) -> None:

        # Get the next time frame
        if len (self._stimuli) > 0:
            time_frame =self._stimuli.pop()
            self._last_time_frame = time_frame
        elif self._remaining_cool_down_duration > 0:
            # If there are no more time frames, hold the last state for the cool down duration
            # Continue the stream
            state = self._hold_function(self._last_time_frame.state)
            self._last_time_frame = TimeFrame(state=state, 
                                              index=self._last_time_frame.index, 
                                              start_time=self._last_time_frame.start_time + self._processing_time, 
                                              duration=self._last_time_frame.duration)
            self._remaining_cool_down_duration -= self._processing_time
        else: # Simulation is over
            return        
        
        # Pass the time frame to the output connections
        for area_index, connection in self._output_connections.items():
            connection.forward(time_frame=time_frame)

    def reset(self) -> None:

        # Call parent
        super().reset()

        # Reset remaining cool down duration
        self._remaining_cool_down_duration = self._cool_down_duration

class StateVisualizer():
    """Superclass for a set of classes that create 2D visualizations of a :py:meth:`.TimeFrame.state` on a 1x1 unit square"""

    def __init__(self):
        pass

    def visualize(self, axes, x, y):
        pass

class BasicAreaTransformation(torch.nn.Module):
    """An :py:class:`.Area` that first averages the time frames for each input buffer, then concatenates them together with the area state along their last axis and finally applies a linear transformation.
    Note:
    
    * The state of the calling :py:class:`.Area` is assumed to be a mere torch.Tensor object, i.e. it is not a list of torch.Tensor objects.
    * The input states of the other :py:class:`.Area` s are assumed to be torch.Tensor objects of shape [instance count, time frame count, dimensionality]. 

    :param input_dimensionality: The dimensionality of the input. This is the sum of the dimensionalities of all inputs (inclduing the state of the calling area).
    :type input_dimensionality: int
    :param output_dimensionality: The dimensionality that the output state shall have.
    :type output_dimensionality: int
    """

    def __init__(self, input_dimensionality: int, output_dimensionality: int) -> "BasicAreaTransformation":
        
        # Call the parent constructor
        super().__init__()

        # Set linear transformation
        self.linear = torch.nn.Linear(input_dimensionality, output_dimensionality)

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
        concatenated = [None] * (len(input_states) + 1)
        concatenated[0] = calling_area_state
        for i, input_state in enumerate(input_states):
            concatenated[i+1] = torch.mean(input_state, dim=Area.TIME_AXIS)
        concatenated = torch.concat(concatenated, dim=-1)
        
        # Apply linear transformation
        return self.linear(concatenated)

class BrIANN(torch.nn.Module):
    """TODO: _summary_

    :param batch_size: The batch size used when passing instances through the areas.
    :type batch_size: int
    :param configuration_file_path: The path to the configuration file.
    :type configuration_file_path: str
    """
    SOURCE_AREA_INDEX = 0
    """The index of the source area in the configuration file. This is used to identify the source area in the configuration file. It is a static constant and set to 0."""
    
    def __init__(self, batch_size: int, configuration_file_path: str) -> "BrIANN":
        
        # Call the parent constructor
        super().__init__()

        # Load the model based on the configuration
        self._load_configuration(batch_size=batch_size, configuration_file_path=configuration_file_path)

        # Set the target area index
        self._TARGET_AREA_INDEX = len(self._areas) - 1

        # Set the time of the simulation
        self._simulation_time = 0.0

        # Set next areas 
        self._next_areas = []

        # Set the flag to indicate that all states are reset
        self._all_states_reset = False


    @property
    def areas(self) -> Dict[int, Area]:
        """A dictionary where each key is an area index and each value is a :py:class:`.Area` object. This is used to access the areas of the model.
        
        :return: The areas of the model.
        :rtype: Dict[int, :py:class:`.Area`]
        """
        return self._areas
    
    @property
    def connection_from(self) -> Dict[int, List[Connection]]:
        """A dictionary where each key is an area index and each value is a list of :py:class:`.Connection` objects that are the output connections of the area with the given index. This is used to access the output connections of the areas of the model.
        
        :return: The output connections of the areas of the model.
        :rtype: Dict[int, List[:py:class:`.Connection`]]
        """
        return self._connection_from
    
    @property
    def connection_to(self) -> Dict[int, List[Connection]]:
        """A dictionary where each key is an area index and each value is a list of :py:class:`.Connection` objects that are the input connections of the area with the given index. This is used to access the input connections of the areas of the model.
        
        :return: The input connections of the areas of the model.
        :rtype: Dict[int, List[:py:class:`.Connection`]]
        """
        return self._connection_to
    
    @property
    def simulation_time(self) -> float:
        """The current simulation time in seconds. This is the time that has passed since the start of the simulation. It is updated after each step of the simulation.
        
        :return: The current simulation time.
        :rtype: float
        """
        return self._simulation_time

    @property
    def TARGET_AREA_INDEX(self) -> int:
        """The index of the target area. This is used to identify the target area in the configuration file. It is set to None before initialization, but it will be automatically set to the largest index occuring in the configuration file during initialization."""
        return self._TARGET_AREA_INDEX

    def _load_configuration(self, batch_size: int, configuration_file_path: str) -> None:
        """Loads the configuration from the given **configuration_file_path** and sets up the areas and connections.
        :param batch_size: The batch size used when passing instances through the areas.
        :type batch_size: int
        :param configuration_file_path: The path to the configuration file.
        :type configuration_file_path: str
        """

        with open(configuration_file_path, "r") as file:
            configuration = json.load(file)
        
        # Find target area index
        self._TARGET_AREA_INDEX = max(item["index"] for item in configuration["areas"])
        if not isinstance(self._TARGET_AREA_INDEX, int):
            raise TypeError("The target area index must be an int.")
        
        # Check if all area indices are integers between the source and target area indices
        area_indices = [item["index"] for item in configuration["areas"]]
        if not all(isinstance(area_index, int) for area_index in area_indices):
            raise TypeError("All area indices must be integers.")
        if not all(area_index in range(self.SOURCE_AREA_INDEX, self._TARGET_AREA_INDEX + 1) for area_index in area_indices):
            raise ValueError("All area indices must be integers between the source area index and the target area index, inclusive.")
        
        # Check validity of batch size
        if not isinstance(batch_size, int):
            raise TypeError("The batch size must be an int.")
        if not batch_size > 0:
            raise ValueError("The batch size must be greater than 0.")

        # Extract initial states from areas to pass to connections
        initial_states = {}
        for item in configuration["areas"]:
            
            # Prepare initial state
            global initial_state
            exec("global initial_state; initial_state = " + item["initial_state"]) # Can be a single tensor or a list of tensors
            
            if isinstance(initial_state, list):
                for state in initial_state:
                    state = torch.concatenate([state[torch.newaxis, :] for _ in range(batch_size)], dim=0) # The batch axis is the first axis
            elif isinstance(initial_state, torch.Tensor):
                initial_state = torch.concatenate([initial_state[torch.newaxis, :] for _ in range(batch_size)], dim=0)
            # No need for an else statement since exceptions will be raised by the area constructor if the initial state is not a tensor or a list of tensors.
                        
            # Store
            initial_states[item["index"]] = initial_state

        # Set connections
        self._connection_from = {area_index: [] for area_index in area_indices}
        self._connection_to = {area_index: [] for area_index in area_indices}
        for item in configuration["connections"]:
            # Extract configuration
            from_area_index = item["from_area_index"]
            to_area_index = item["to_area_index"]
            global transformation
            exec("global transformation; transformation = " + item["transformation"])
            initial_time_frame = TimeFrame(state=initial_states[from_area_index], index=-1, start_time=-1.0, duration=1.0)
            connection = Connection(from_area_index=from_area_index, to_area_index=to_area_index, initial_time_frame=initial_time_frame, transformation=transformation)
            
            # Insert the connection to the from array
            if from_area_index not in self._connection_from: self._connection_from[from_area_index] = []
            self._connection_from[from_area_index].append(connection)

            # Insert the output connection to the to array
            if to_area_index not in self._connection_to: self._connection_to[to_area_index] = []
            self._connection_to[to_area_index].append(connection)

        # Set areas
        self._areas = {}
        for item in configuration["areas"]:
            
            # Enrich configuration
            area_index = item["index"]
            
            if "hold_function" in item.keys(): 
                global hold_function
                exec("global hold_function; hold_function = " + item["hold_function"])
                item["hold_function"] = hold_function
            
            item["output_connections"] = {connection.to_area_index : connection for connection in self._connection_from[area_index]} 
            
            # Prepare initial state and output connections
            item["initial_state"] = initial_states[item["index"]]
            item["output_connections"] = {connection.to_area_index : connection for connection in self._connection_from[area_index]}
            
            # Create source
            if area_index == self.SOURCE_AREA_INDEX: 
                self._areas[self.SOURCE_AREA_INDEX] = Source(**item)
            # Create other areas
            else:
                
                # Prepare other parameters
                item["input_time_frame_buffers"] = {connection.from_area_index : connection.output_time_frame_buffer for connection in self._connection_to[area_index]}
                if "transformation" in item.keys():
                    exec("global transformation; transformation = " + item["transformation"])
                    item["transformation"] = transformation
                
                # Create area
                self._areas[area_index] = Area(**item)

        # Set flag to indicate that all states are reset
        self._all_states_reset = True

    def start(self, stimuli: Deque[TimeFrame]) -> None:
        """Starts the simulation of a trial with the given **stimuli** as input.

        :param stimuli: The :py:class:`.TimeFrame` objects to be be processed by the model. The time frames have to be ordered within the deque such that they will get extacted chronologically by the deque.pop method. The :py:meth:`~.TimeFrame.state` of each time frame is required to enumerate instances along the :py:attr:`.Area.BATCH_AXIS`. **IMPORTANT**: The duration of each time frame must be equal to the :py:meth:`~.Area.processing_time` of the source area, in seconds. 
        :type stimuli: Deque[:py:class:`.TimeFrame`]
        :raises ValueError: if the duration of any time frame is not equal to the :py:meth:`~.Area.processing_time` of the source area, in seconds.
        """
        
        # Reset the states of all areas
        if not self._all_states_reset:
            for area in self._areas.values():
                area.reset()
            self._all_states_reset = True

        # Provide stimuli to the source area
        self._areas[self.SOURCE_AREA_INDEX].load_stimuli(stimuli=stimuli)

    def step(self) -> TimeFrame:
        """Performs one step of the simulation by processing the next :py:class:`.TimeFrame` from the input stimuli and passing it through the areas.
        This method needs to be called repeatedly until the end of the simulation.

        :raises Exception: if the simulation is over.
        :return: The current time frame output by the target area, in case the target area was due in the current step. Otherwise, None is return
        :rtype: :py:class:`.TimeFrame`
        """
        
        # If no more areas are due, stop the simulation
        if self._areas[self.SOURCE_AREA_INDEX].remaining_cool_down_duration <= 0:            
            raise Exception("Cannot take another step because simulation is over.")

        # Find the areas that are due next
        self._next_areas = []
        min_time = sys.float_info.max
        for area in self._areas.values():
            area_next_time = (area.produced_time_frame_count+1) * area.processing_time # Add 1 to get the time of the area's next frame 
            if abs(area_next_time - min_time) < 1e-10: # Current area belongs to current set of next areas, accounting for small numerical timing errors
                self._next_areas.append(area)
            elif area_next_time < min_time - 1e-10: # Current area is due sooner 
                self._next_areas = [area]
                min_time = area_next_time
        
        # Propagate
        for area in self._next_areas: area.forward()

        # Update the simulation time
        self._simulation_time += min_time

        # Extract latest time frame
        target_area = self._areas[self.TARGET_AREA_INDEX]
        if target_area in self._next_areas:
            new_time_frame = TimeFrame(state=target_area.state, 
                                       index=target_area.produced_time_frame_count-1, # Subtract 1, since the target area already incremeted its counter during the forward call
                                       start_time=target_area.produced_time_frame_count * target_area.processing_time, 
                                       duration=target_area.processing_time)
            return new_time_frame
        else:
            return None
        
    def plot_graph(self, axes: plt.Axes = None) -> List:
        """Plots the areas and connections of the model using a circular layout.

        :param axes: In case the plot should be placed on a particular pair of axes, they can be provided here.
        :type axes: matplotlib.pyplot.Axes, optional, defaults to None
        :return artists: A list of artists, or artist lists or artist dictionaries that have been created during plotting.
        :rtype: List
        """

        # Create a directed graph
        G = nx.DiGraph()
        node_colors = []
        
        # Add nodes for each area
        for area in self._areas.values():
            G.add_node(area.index, label=area.index)
            node_colors.append('orange' if area in self._next_areas else 'lightgray')
        
        # Add edges for each connection
        edge_list = []
        for from_area_index, connections in self._connection_from.items():
            for connection in connections:
                edge_list.append((from_area_index, connection.to_area_index, {'length': 1}))
        G.add_edges_from(edge_list)

        # Draw the graph
        if axes == None:
            plt.figure(figsize=(15,10))
            axes = self.fig.add_subplot()
            
        # Graph
        area_positions = nx.circular_layout(G)
        scatter = nx.draw_networkx_nodes(G, area_positions, ax=axes, node_color=node_colors, node_size=800, alpha= 0.9)
        node_labels = nx.draw_networkx_labels(G, area_positions, ax=axes)
        
        curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))
        straight_edges = nx.draw_networkx_edges(G, area_positions, ax=axes, edgelist=straight_edges, arrows=True, node_size=800, alpha= 0.9)
        curved_edges = nx.draw_networkx_edges(G, area_positions, ax=axes, edgelist=curved_edges, connectionstyle=f'arc3, rad = {0.25}', arrows=True, node_size=800, alpha= 0.9)

        edge_labels = dict([((u, v,), f'{d["length"]}\n\n{G.edges[(v,u)]["length"]}') for u, v, d in G.edges(data=True) if area_positions[u][0] > area_positions[v][0]])
        edge_labels = nx.draw_networkx_edge_labels(G, area_positions, edge_labels=edge_labels, font_color='red')

        # Area data
        #for area_index, position in area_positions.items():
        #    axes.text(x=1.2*position[0], y=1.2*position[1], s='Hello')

        return [scatter, node_labels, straight_edges, curved_edges, edge_labels]
        """
        # Adjacency
        plt.subplot(1,2,2)
        adjacency_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
        plt.imshow(adjacency_matrix, cmap='Blues', interpolation='nearest')
        #plt.colorbar(label='Connection Strength')
        plt.xlabel("To Area Index")
        plt.ylabel("From Area Index")

        plt.show()
        """

    def __repr__(self) -> str:
        string = "BrIANN\n"

        for area in self._areas.values(): 
            string += f"{area}\n"
            for connection in self._connection_from[area.index]:
                string += f"\t{connection}\n"

        return string


