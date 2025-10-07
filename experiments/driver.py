from typing import Iterator, List, Dict, Any, Tuple
import random, numpy as np
import json, sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.utilities import file_management as bpufm

# Generate briann configuration
def generate_briann_configuration(network_configuration, connection_configurations, area_configurations, dataset_configurations) -> Dict[str, Any]:
    configuration = {
        "network": network_configuration,
        "connections": connection_configurations,
        "areas": area_configurations,
        "datasets": dataset_configurations
    }

    return configuration

def generate_regular_area_configurations(regular_area_count: int, 
                                        index_iterator: Iterator[int],
                                        initial_state_iterator: Iterator[str],
                                        input_shape_iterator: Iterator[List[int]],
                                        output_shape_iterator: Iterator[List[int]],
                                        update_rate_iterator: Iterator[float], 
                                        transformation_iterator: Iterator[str]) -> List[Dict[str,Any]]:
    
    # Start with sources and targets
    area_configurations = []
    
    # Add regular areas
    for _ in range(regular_area_count):
        area_configuration = {
            "index": next(index_iterator),
            "initial_state": next(initial_state_iterator),
            "input_shape": next(input_shape_iterator),
            "output_shape": next(output_shape_iterator),
            "update_rate": next(update_rate_iterator),
            "transformation": next(transformation_iterator)
        }
        area_configurations.append(area_configuration)

    # Return
    return area_configurations

def index_iterator(start: int = 0) -> Iterator[int]:
    i = start
    while True:
        yield i
        i += 1

def initial_state_iterator(shape: List[int]) -> Iterator[str]:
    while True:
        yield f"torch.zeros({shape})"

def shape_iterator(shape: List[int]) -> Iterator[List[int]]:
    while True:
        yield shape

def update_rate_iterator(update_rate: float) -> Iterator[float]:
    while True:
        yield update_rate

def area_transformation_iterator(area_output_dimensionality: int, area_input_dimensionality: int) -> Iterator[str]:
    while True:
        yield f"AreaTransformation.LinearRecursive(area_output_dimensionality={area_output_dimensionality}, area_input_dimensionality={area_input_dimensionality})"

def generate_adjacency_matrix(source_indices: List[int], target_indices: List[int], regular_area_count, connection_probability, seed=None):
    
    total_area_count = len(source_indices) + regular_area_count + len(target_indices)
    adjacency_matrix = np.zeros((total_area_count, total_area_count), dtype=int)
    random.seed(seed)

    # Iterate possible connections
    for i in range(total_area_count):
        for j in range(total_area_count): 
            
            # Check eligibitly
            if i in target_indices or j in source_indices:
                continue

            # Create connection
            adjacency_matrix[i, j] = random.random() < connection_probability
                
    return adjacency_matrix

def area_indices_to_connection(connection_index: int, from_area_index: int, to_area_index: int, recurrent_area_indices: List[int], area_configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    # Get area configurations
    from_area_configuration = next(filter(lambda area_configuration: area_configuration["index"] == from_area_index, area_configurations))
    to_area_configuration = next(filter(lambda area_configuration: area_configuration["index"] == to_area_index, area_configurations))
    a = from_area_configuration["output_shape"][-1]
    b = to_area_configuration["input_shape"][-1]
    c = to_area_configuration["output_shape"][-1]

    # Seperate slots for recurrent and other connections
    if to_area_index in recurrent_area_indices: 
        
        # Self connection (i.e. recurrent)
        if from_area_index == to_area_index:
            transformation = f"ConnectionTransformation.MoveToSlot(from_area_output_dimensionality={a}, to_area_input_dimensionality={b}, to_area_output_dimensionality={c}, slot='self'))"
            
        # Other connection (i.e. signal transmitted between distinct areas)
        else:
            linear = f"torch.nn.Linear(in_features={a}, out_features={b-c}, bias=False)"
            move_to_slot = f"ConnectionTransformation.MoveToSlot(from_area_output_dimensionality={a}, to_area_input_dimensionality={b}, to_area_output_dimensionality={c}, slot='other'))"
            transformation = f"torch.nn.Sequential({linear}, {move_to_slot})"
    
    else:
        transformation = f"torch.nn.Linear(in_features={a}, out_features={b}, bias=False)"
        if from_area_index == 4 and to_area_index == 1:
            tmp = 1

    # Create connection configuration
    connection_configuration = {
        "index": connection_index,
        "from_area_index": from_area_index,
        "to_area_index": to_area_index,
        "transformation": transformation
    }

    # Output
    return connection_configuration

def adjacency_matrix_to_connections(adjacency_matrix: np.ndarray, source_area_indices: List[int], target_area_indices: List[int], area_configurations: List[Dict[str,Any]], area_indices_to_connection: callable):
    connections = []
    area_count = adjacency_matrix.shape[0]
    recurrent_area_indices = [area_index for area_index in range(area_count) if adjacency_matrix[area_index,area_index]]
    connection_index = 0
    
    # Iterate matrix
    for i in range(area_count):
        for j in range(area_count):
            
            # Check whether there is a connection from area i to j
            if adjacency_matrix[i, j]:
            
                # Create connection
                connections.append(area_indices_to_connection(connection_index = connection_index, 
                                                            from_area_index = i, to_area_index =j, 
                                                            recurrent_area_indices = recurrent_area_indices, 
                                                            area_configurations = area_configurations))
                
                connection_index += 1

    return connections

if __name__ == "__main__":
    
    # Load training configuration
    path = bpufm.map_path_to_os(path="experiments/training_configuration.json")
    with open(path, 'r') as file:
        configuration = json.loads(file.read())

    # Generate area configurations
    regular_area_configurations = generate_regular_area_configurations(
        regular_area_count = configuration["regular_area_count"], 
        index_iterator = index_iterator(start=len(configuration["source_configurations"]) + len(configuration["target_configurations"])),
        initial_state_iterator = initial_state_iterator(shape=configuration["regular_area_output_shape"]),
        input_shape_iterator = shape_iterator(shape=configuration["regular_area_input_shape"]),
        output_shape_iterator = shape_iterator(shape=configuration["regular_area_output_shape"]),
        update_rate_iterator = update_rate_iterator(update_rate=configuration["regular_area_update_rate"]), 
        transformation_iterator = area_transformation_iterator(area_output_dimensionality=configuration["regular_area_output_shape"][-1], area_input_dimensionality=configuration["regular_area_input_shape"][-1]))
    
    area_configurations = configuration["source_configurations"] + configuration["target_configurations"] + regular_area_configurations

    # Generate connection configurations
    source_indices = [area_configuration["index"] for area_configuration in configuration["source_configurations"]]
    target_indices = [area_configuration["index"] for area_configuration in configuration["target_configurations"]]
    
    adjacency_matrix = generate_adjacency_matrix(
        source_indices = source_indices, 
        target_indices = target_indices, 
        regular_area_count = len(regular_area_configurations), 
        connection_probability = configuration["connection_probability"], 
        seed=configuration["random_seed"])
    
    connection_configurations = adjacency_matrix_to_connections(
        adjacency_matrix = adjacency_matrix,
        source_area_indices = [area_configuration["index"] for area_configuration in configuration["source_configurations"]],
        target_area_indices = [area_configuration["index"] for area_configuration in configuration["target_configurations"]],
        area_configurations = area_configurations,
        area_indices_to_connection = area_indices_to_connection)
    
    # Create briann configuration
    briann_configuration = generate_briann_configuration(
        network_configuration=configuration["network"],
        connection_configurations=connection_configurations,
        area_configurations=area_configurations,
        dataset_configurations=configuration["datasets"])

    print(json.dumps(briann_configuration, indent=4))
    
    # TODO: Ensure random seed is used correctly