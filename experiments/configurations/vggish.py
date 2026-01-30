import torch, sys, os
sys.path.append(os.path.abspath(""))
from briann.network import area_transformations as bpnat
from briann.network import connection_transformations as bpnct
from briann.network import core as bpnc

from briann.training import data_management as bptdm
from briann.utilities import callbacks as bpuc
from briann.utilities import core as bpuco
import numpy as np
from typing import Set

class Transpose(torch.nn.Module):
    def __init__(self, first_axis: int, second_axis: int):
        super().__init__()
        self.first_axis, self.second_axis = first_axis, second_axis

    def forward(self, x):
        return torch.transpose(x, self.first_axis, self.second_axis)

class Reshape(torch.nn.Module):
    def __init__(self, new_shape: torch.Size):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return torch.reshape(x, shape=[-1] + self.new_shape)

# Timeframe accumulators
_batch_size = 16
_time_frame_accumulators = {}
for i, dimensionality in enumerate([[1, 96, 64], [64, 96, 64], [64, 48, 32], [128, 48, 32], [128, 24, 16], [256, 24, 16], [256, 24, 16], [256, 12, 8], [512, 12, 8], [512, 12, 8], [512, 6, 4], [4096], [4096], [128], [128]]):
    _time_frame_accumulators[i] = bpnc.TimeFrameAccumulator(initial_time_frame=bpnc.TimeFrame(state=torch.zeros([_batch_size] + dimensionality), time_point=0.0), decay_rate=1.0) 

# Set connections
connection_list = [None] * 14
for i in range(len(connection_list)):
    connection_list[i] = bpnc.Connection(index=i, from_area_index=i, to_area_index=i+1, input_time_frame_accumulator=_time_frame_accumulators[i], transformation=torch.nn.Identity())

connection_list.append(bpnc.Connection(index=14, from_area_index=13, to_area_index=7, input_time_frame_accumulator=_time_frame_accumulators[13], transformation=torch.nn.Sequential(
    torch.nn.Linear(in_features=128, out_features=256*24*16),
    Reshape(new_shape=[256, 24,16])
)))
connection_list.append(bpnc.Connection(index=15, from_area_index=7, to_area_index=1, input_time_frame_accumulator=_time_frame_accumulators[7], transformation=torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=256*12*8, out_features=1*96*64),
    Reshape(new_shape=[1, 96, 64])
)))

_connections = torch.nn.ModuleList(*[connection_list])

def get_connections_from(connections, area_index: int) -> Set[bpnc.Connection]:
    
    # Compile
    result = [None] * len(connections)
    i = 0
    for connection in connections:
        if connection.from_area_index == area_index: 
            result[i] = connection
            i += 1

    # Output
    return set(result[:i])
    
def get_connections_to(connections, area_index: int) -> Set[bpnc.Connection]:

    # Compile
    result = [None] * len(connections)
    i = 0
    for connection in connections:
        if connection.to_area_index == area_index: 
            result[i] = connection
            i += 1

    # Output
    return set(result[:i])

# Set areas
_areas = []

## Source area
_areas.append(bpnc.Source(index=0, 
                             output_time_frame_accumulator=_time_frame_accumulators[0], 
                             output_shape=[1, 96, 64], 
                             output_connections=get_connections_from(connections=_connections, area_index=0), 
                             update_rate=1.0))

# Regular areas
# First Block
_areas.append(bpnc.Area(index=1,
                        output_time_frame_accumulator=_time_frame_accumulators[1],
                        input_connections=get_connections_to(connections=_connections, area_index=1),
                        input_shape=[1, 96, 64],
                        output_shape=[64, 96, 64],
                        output_connections=get_connections_from(connections=_connections, area_index=1),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
                            torch.nn.ReLU(inplace=True)),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=2,
                        output_time_frame_accumulator=_time_frame_accumulators[2],
                        input_connections=get_connections_to(connections=_connections, area_index=2),
                        input_shape=[64, 96, 64],
                        output_shape=[64, 48, 32],
                        output_connections=get_connections_from(connections=_connections, area_index=2),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.MaxPool2d(kernel_size=2, stride=2),
                        update_rate=1.0))
    
# Second Block
_areas.append(bpnc.Area(index=3,
                        output_time_frame_accumulator=_time_frame_accumulators[3],
                        input_connections=get_connections_to(connections=_connections, area_index=3),
                        input_shape=[64, 48, 32],
                        output_shape=[128, 48, 32],
                        output_connections=get_connections_from(connections=_connections, area_index=3),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                            torch.nn.ReLU(inplace=True)),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=4,
                        output_time_frame_accumulator=_time_frame_accumulators[4],
                        input_connections=get_connections_to(connections=_connections, area_index=4),
                        input_shape=[128, 48, 32],
                        output_shape=[128, 24, 16],
                        output_connections=get_connections_from(connections=_connections, area_index=4),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.MaxPool2d(kernel_size=2, stride=2),
                        update_rate=1.0))

# Third Block
_areas.append(bpnc.Area(index=5,
                        output_time_frame_accumulator=_time_frame_accumulators[5],
                        input_connections=get_connections_to(connections=_connections, area_index=5),
                        input_shape=[128, 24, 16],
                        output_shape=[256, 24, 16],
                        output_connections=get_connections_from(connections=_connections, area_index=5),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                            torch.nn.ReLU(inplace=True)),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=6,
                        output_time_frame_accumulator=_time_frame_accumulators[6],
                        input_connections=get_connections_to(connections=_connections, area_index=6),
                        input_shape=[256, 24, 16],
                        output_shape=[256, 24, 16],
                        output_connections=get_connections_from(connections=_connections, area_index=6),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                            torch.nn.ReLU(inplace=True)),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=7,
                        output_time_frame_accumulator=_time_frame_accumulators[7],
                        input_connections=get_connections_to(connections=_connections, area_index=7),
                        input_shape=[256, 24, 16],
                        output_shape=[256, 12, 8],
                        output_connections=get_connections_from(connections=_connections, area_index=7),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.MaxPool2d(kernel_size=2, stride=2),
                        update_rate=1.0))

# Fourth Block
_areas.append(bpnc.Area(index=8,
                        output_time_frame_accumulator=_time_frame_accumulators[8],
                        input_connections=get_connections_to(connections=_connections, area_index=8),
                        input_shape=[256, 12, 8],
                        output_shape=[512, 12, 8],
                        output_connections=get_connections_from(connections=_connections, area_index=8),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                            torch.nn.ReLU(inplace=True)),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=9,
                        output_time_frame_accumulator=_time_frame_accumulators[9],
                        input_connections=get_connections_to(connections=_connections, area_index=9),
                        input_shape=[512, 12, 8],
                        output_shape=[512, 12, 8],
                        output_connections=get_connections_from(connections=_connections, area_index=9),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                            torch.nn.ReLU(inplace=True)),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=10,
                        output_time_frame_accumulator=_time_frame_accumulators[10],
                        input_connections=get_connections_to(connections=_connections, area_index=10),
                        input_shape=[512, 12, 8],
                        output_shape=[512, 6, 4],
                        output_connections=get_connections_from(connections=_connections, area_index=10),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.MaxPool2d(kernel_size=2, stride=2),
                        update_rate=1.0))
    
# Reshaping block
_areas.append(bpnc.Area(index=11,
                        output_time_frame_accumulator=_time_frame_accumulators[11],
                        input_connections=get_connections_to(connections=_connections, area_index=11),
                        input_shape=[512, 6, 4],
                        output_shape=[4096],
                        output_connections=get_connections_from(connections=_connections, area_index=11),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            Transpose(first_axis=1, second_axis=3),
                            Transpose(first_axis=1, second_axis=2),
                            torch.nn.Flatten(),
                            torch.nn.Linear(in_features=12288, out_features=4096),
                            torch.nn.ReLU(inplace=True)
                        ),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=12,
                        output_time_frame_accumulator=_time_frame_accumulators[12],
                        input_connections=get_connections_to(connections=_connections, area_index=12),
                        input_shape=[4096],
                        output_shape=[4096],
                        output_connections=get_connections_from(connections=_connections, area_index=12),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Linear(in_features=4096, out_features=4096),
                            torch.nn.ReLU(inplace=True)
                        ),
                        update_rate=1.0))

_areas.append(bpnc.Area(index=13,
                        output_time_frame_accumulator=_time_frame_accumulators[13],
                        input_connections=get_connections_to(connections=_connections, area_index=13),
                        input_shape=[4096],
                        output_shape=[128],
                        output_connections=get_connections_from(connections=_connections, area_index=13),
                        merger=bpnc.AdditiveMerger(),
                        transformation=torch.nn.Sequential(
                            torch.nn.Linear(in_features=4096, out_features=128),
                            torch.nn.ReLU(inplace=True)
                        ),
                        update_rate=1.0))

# Target area
_areas.append(bpnc.Target(index=14, 
                 output_time_frame_accumulator=_time_frame_accumulators[14], 
                 input_connections=get_connections_to(connections=_connections, area_index=14),
                 input_shape=[128],
                 output_shape=[128],
                 merger=bpnc.AdditiveMerger(),
                 transformation=torch.nn.Identity(), 
                 update_rate=1.0))

# Load weights for all areas

# Construct briann
_briann = bpnc.BrIANN(name="VGGish", areas=_areas, connections=_connections)

sinusoids = bptdm.Sinusoids(instance_count=100, duration=10.0, sampling_rate=16000, frequency_count=3, noise_range=0.5)
_data_iterator = iter(torch.utils.data.DataLoader(dataset=bptdm.VGGishSpectrograms(sinusoids=sinusoids), 
                                                  batch_size=_batch_size, 
                                                  drop_last=True))

# Training configuration
training_configuration = {
    "model": _briann,
    "data_iterator": _data_iterator,
    "input_shape": [1, 96, 64],
    "output_shape": [128],
    "class_count": 1,
    "loss_function": torch.nn.CrossEntropyLoss(),
    "optimizer": torch.optim.SGD(_briann.parameters(), lr=0.01, momentum=0.9),
    "epoch_count": 5,
    "steps_per_batch": 10,
}

# Inference configuration
inference_configuration = {
    "model": _briann,
    "data_iterator": _data_iterator,
    "input_shape": [1, 96, 64],
    "output_shape": [128],
    "class_count": 1,
    "steps_per_batch": 10,
}