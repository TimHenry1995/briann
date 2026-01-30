import torch, sys, os
sys.path.append(os.path.abspath(""))
from briann.network import area_transformations as bnat
from briann.network import connection_transformations as bnct
from briann.network import core as bnc

from experiments.sinusoids import data_loader as esdl

from typing import Set

# Timeframe accumulators
_batch_size = 16
_time_frame_accumulators = {}
for _i, _dimensionality in enumerate([1,7,4,2,1,3]):
    _time_frame_accumulators[_i] = bnc.TimeFrameAccumulator(initial_time_frame=bnc.TimeFrame(state=torch.zeros([_batch_size,_dimensionality]), time_point=0.0), decay_rate=0.1) 

# Set connections
_connections = torch.nn.ModuleList([
    bnc.Connection(index=0, from_area_index=0, to_area_index=1, input_time_frame_accumulator=_time_frame_accumulators[0], transformation=torch.nn.Linear(1,3, bias=False)),
    bnc.Connection(index=1, from_area_index=1, to_area_index=2, input_time_frame_accumulator=_time_frame_accumulators[1], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[0,1], output_flatten_axes=(1,1), output_shape=[2]), torch.nn.Linear(2,5, bias=False))),
    bnc.Connection(index=2, from_area_index=2, to_area_index=2, input_time_frame_accumulator=_time_frame_accumulators[2], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[0,1], output_flatten_axes=(1,1), output_shape=[2]), torch.nn.Linear(2,5, bias=False))),
    bnc.Connection(index=3, from_area_index=2, to_area_index=4, input_time_frame_accumulator=_time_frame_accumulators[2], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[2,3], output_flatten_axes=(1,1), output_shape=[2]), torch.nn.Linear(2,3, bias=False))),
    bnc.Connection(index=4, from_area_index=1, to_area_index=1, input_time_frame_accumulator=_time_frame_accumulators[1], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[5,6], output_flatten_axes=(1,1), output_shape=[2]), torch.nn.Linear(2,3, bias=False))),
    bnc.Connection(index=5, from_area_index=1, to_area_index=3, input_time_frame_accumulator=_time_frame_accumulators[1], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[2,3,4], output_flatten_axes=(1,1), output_shape=[3]), torch.nn.Linear(3,2, bias=False))),
    bnc.Connection(index=6, from_area_index=3, to_area_index=3, input_time_frame_accumulator=_time_frame_accumulators[3], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[0,1], output_flatten_axes=(1,1), output_shape=[2]), torch.nn.Linear(2,2, bias=False))),
    bnc.Connection(index=7, from_area_index=3, to_area_index=4, input_time_frame_accumulator=_time_frame_accumulators[3], transformation=torch.nn.Sequential(bnct.IndexBasedSplitter(input_flatten_axes=(1,1), input_indices=[0,1], output_flatten_axes=(1,1), output_shape=[2]), torch.nn.Linear(2,2, bias=False))),
    bnc.Connection(index=8, from_area_index=4, to_area_index=5, input_time_frame_accumulator=_time_frame_accumulators[4], transformation=torch.nn.Linear(1,2, bias=False))
])


def get_connections_from(connections, area_index: int) -> Set[bnc.Connection]:
    
    # Compile
    result = [None] * len(connections)
    i = 0
    for connection in connections:
        if connection.from_area_index == area_index: 
            result[i] = connection
            i += 1

    # Output
    return set(result[:i])
    
def get_connections_to(connections, area_index: int) -> Set[bnc.Connection]:

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
_areas.append(bnc.Source(index=0, 
                             output_time_frame_accumulator=_time_frame_accumulators[0], 
                             output_shape=[1], 
                             output_connections=get_connections_from(connections=_connections, area_index=0), 
                             update_rate=10))

# Regular areas
_areas.append(bnc.Area(index=1,
                         output_time_frame_accumulator=_time_frame_accumulators[1],
                         input_connections=get_connections_to(connections=_connections, area_index=1),
                         input_shape=[3],
                         output_shape=[7],
                         output_connections=get_connections_from(connections=_connections, area_index=1),
                         merger=bnc.AdditiveMerger(),
                         transformation=bnat.SimpleDenseTransformation(input_dimensionality=3, output_dimensionality=7),
                         update_rate=9.0))

_areas.append(bnc.Area(index=2,
                         output_time_frame_accumulator=_time_frame_accumulators[2],
                         input_connections=get_connections_to(connections=_connections, area_index=2),
                         input_shape=[5],
                         output_shape=[4],
                         output_connections=get_connections_from(connections=_connections, area_index=2),
                         merger=bnc.AdditiveMerger(),
                         transformation=bnat.SimpleDenseTransformation(input_dimensionality=5, output_dimensionality=4),
                         update_rate=8.0))

_areas.append(bnc.Area(index=3,
                         output_time_frame_accumulator=_time_frame_accumulators[3],
                         input_connections=get_connections_to(connections=_connections, area_index=3),
                         input_shape=[4],
                         output_shape=[2],
                         output_connections=get_connections_from(connections=_connections, area_index=3),
                         merger=bnc.IndexBasedMerger(connection_index_to_input_flatten_axes={5:(1,1),6:(1,1)}, connection_index_to_output_indices={5:[0,1], 6:[2,3]}, output_flatten_axes=(1,1), final_output_shape=[4]),
                         transformation=bnat.SimpleDenseTransformation(input_dimensionality=4, output_dimensionality=2),
                         update_rate=7.0))

_areas.append(bnc.Area(index=4,
                         output_time_frame_accumulator=_time_frame_accumulators[4],
                         input_connections=get_connections_to(connections=_connections, area_index=4),
                         input_shape=[5],
                         output_shape=[1],
                         output_connections=get_connections_from(connections=_connections, area_index=4),
                         merger=bnc.IndexBasedMerger(connection_index_to_input_flatten_axes={3:(1,1), 7:(1,1)}, connection_index_to_output_indices={3:[0,1,2], 7:[3,4]} , output_flatten_axes=(1,1), final_output_shape=[5]),
                         transformation=bnat.SimpleDenseTransformation(input_dimensionality=5, output_dimensionality=1),
                         update_rate=6.0))


# Target area
_areas.append(bnc.Target(index=5, 
                 output_time_frame_accumulator=_time_frame_accumulators[5], 
                 input_connections=get_connections_to(connections=_connections, area_index=5),
                 input_shape=[2],
                 output_shape=[3],
                 merger=bnc.AdditiveMerger(),
                 transformation=torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Softmax(dim=-1)), 
                 update_rate=5.0))

# Construct briann
_briann = bnc.BrIANN(name="Briann 2", areas=_areas, connections=_connections)

# Data loader
_data_iterator = iter(torch.utils.data.DataLoader(dataset=esdl.Sinusoids(instance_count=100, duration=10.0, sampling_rate=10, frequency_count=3, noise_range=0.0), 
                                                  batch_size=_batch_size, 
                                                  drop_last=True))

# Training configuration
training_configuration = {
    "model": _briann,
    "data_iterator": _data_iterator,
    "input_shape": [1],
    "output_shape": [1],
    "class_count": 3,
    "loss_function": torch.nn.CrossEntropyLoss(),
    "optimizer": torch.optim.SGD(_briann.parameters(), lr=0.01, momentum=0.9),
    "epoch_count": 5,
    "steps_per_batch": 10,
}

# Inference configuration
inference_configuration = {
    "model": _briann,
    "data_iterator": _data_iterator,
    "input_shape": [1],
    "output_shape": [1],
    "class_count": 3,
    "steps_per_batch": 10,
}