from typing import Iterator, List, Dict, Any
import random, numpy as np

# Generate briann configuration

def generate_briann_configuration(name, decay_rate, batch_size, , area_configurations):
    configuration = {
        "network":
            {
                "name": name,
                "decay_rate": decay_rate,
                "batch_size": batch_size
            },
    }

def adjacency_matrix_to_connections(adjacency_matrix, area_indices_to_connections: callable):
    connections = []
    num_areas = adjacency_matrix.shape[0]

    for i in range(num_areas):
        for j in range(num_areas):
            
            # Check whether there is a connection from area i to j
            if adjacency_matrix[i, j] == 0 or adjacency_matrix[i, j] == None:
                
                if adjacency_matrix[i, j]:
                
                    connections += area_indices_to_connections(i,j)

    return connections


def generate_area_configurations(source_configurations: List[Dict[str,Any]], target_configurations: List[Dict[str,Any]],
                                 regular_area_count: int, 
                                 index_iterator: Iterator[int],
                                 initial_state_iterator: Iterator[str],
                                 update_rate_iterator: Iterator[float], 
                                 state_merge_strategy_iterator: Iterator[str], 
                                 transformation_iterator: Iterator[str]) -> List[Dict[str,Any]]:
    
    # Start with sources and targets
    area_configurations = source_configurations + target_configurations
    
    # Add regular areas
    for _ in range(regular_area_count):
        area_configuration = {
            "index": next(index_iterator),
            "initial_state": next(initial_state_iterator),
            "update_rate": next(update_rate_iterator),
            "state_merge_strategy": next(state_merge_strategy_iterator),
            "transformation": next(transformation_iterator)
        }
        area_configurations.append(area_configuration)

    # Return
    return area_configurations

def generate_adjacency_matrix(source_count: int, target_count: int, regular_area_count, connection_probability, allow_self_connections, seed=None):
    
    total_area_count = source_count + regular_area_count + target_count
    adjacency_matrix = np.zeros((total_area_count, total_area_count), dtype=bool)
    random.seed(seed)

    for i in range(total_area_count - target_count): # Connection start from sources and regular areas only
        for j in range(source_count, total_area_count): # Connections end in regular areas and targets only
            
            # Handle self-connections
            if i == j and not allow_self_connections:
                continue
            
            # Create connection
            adjacency_matrix[i, j] = random.random() < connection_probability
                
    return adjacency_matrix


# Load training configuration
def load_training_configuration():
    pass

# Execute training