# https://edwin-de-jong.github.io/blog/mnist-sequence-data/
import sys
import os
sys.path.append(os.path.abspath(""))
from briann.python.utilities import random as bpur
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from src.briann.python.utilities import utilities as bpuu

def collate_function(sequences: List[Tuple[torch.Tensor, torch.Tensor]], **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to pad sequences to the same length.
    
    :param sequences: The sequences to collate.
    :type sequences: List[Tuple[torch.Tensor, torch.Tensor]]
    :return: The collated sequences.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    # Unzip sequences
    X, y = zip(*sequences)

    # Pad X
    X = torch.nn.utils.rnn.pad_sequence(sequences=X, **kwargs)

    # Stack y
    y = torch.stack(y)

    return X, y

class Sinusoids(Dataset):
    def __init__(self, instance_count: int, duration: float, sampling_rate: float, frequency_count: int, noise_range: float, *args, **kwargs) -> "Sinusoids":
        # Call super
        super().__init__(*args, **kwargs)

        # Check input validity
        if not isinstance(instance_count, int): raise TypeError(f"The instance_count was expected to be an integer but is {type(instance_count)}.")
        if instance_count <= 0: raise ValueError(f"The instance_count was expected to be a positive integer but is {instance_count}.")

        if not isinstance(duration, (int, float)): raise TypeError(f"The duration was expected to be a number but is {type(duration)}.")
        if duration <= 0: raise ValueError(f"The duration was expected to be a positive number but is {duration}.")

        if not isinstance(sampling_rate, (int, float)): raise TypeError(f"The sampling_rate was expected to be a number but is {type(sampling_rate)}.")
        if sampling_rate <= 0: raise ValueError(f"The sampling_rate was expected to be a positive number but is {sampling_rate}.")

        if not isinstance(frequency_count, int): raise TypeError(f"The frequency_count was expected to be an integer but is {type(frequency_count)}.")
        if frequency_count <= 0: raise ValueError(f"The frequency_count was expected to be a positive integer but is {frequency_count}.")

        if not isinstance(noise_range, (int, float)): raise TypeError(f"The noise_range was expected to be a number but is {type(noise_range)}.")
        if noise_range < 0: raise ValueError(f"The noise_range was expected to be a non-negative number but is {noise_range}.")

        # Set attributes
        self._instance_count = instance_count
        self._duration = (float)(duration)
        self._sampling_rate = sampling_rate
        self._frequency_count = frequency_count
        self._noise_range = noise_range

    @property
    def instance_count(self) -> int:
        return self._instance_count
    
    @property
    def duration(self) -> int:
        return self._duration
    
    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate
    
    @property
    def frequency_count(self) -> int:
        return self._frequency_count
    
    @property
    def noise_range(self) -> float:
        return self._noise_range

    def __len__(self):
        return self.instance_count
    
    def __getitem__(self, idx):
        # Choose frequency
        frequency = idx % self._frequency_count + 1
        y = frequency

        # Generate sinusoid
        X = torch.zeros([(int)(self._sampling_rate * self._duration), 2])
        X[:,0] = torch.linspace(start=0.0, end=self._duration, steps=(int)(self._sampling_rate * self._duration))
        X[:,1] = torch.sin(X[:,0] * 2 * torch.pi * frequency)
        noise = torch.Tensor(bpur.StatelessRandom.pseudo_uniform(lower_bound=-self._noise_range/2, upper_bound=self._noise_range/2, seed=idx, count=X.shape[0]))
        X[:,1] += noise

        # Output
        return X, y

class PenStrokeMNIST(Dataset):
    
    def __init__(self, folder_path: str) -> "PenStrokeMNIST":
        # Call super
        super().__init__()

        # Set attributes
        self.folder_path = folder_path
        
    @property
    def folder_path(self) -> str:
        """The path to the folder where the data is stored.

        :return: The path to the folder where the data is stored.
        :rtype: str
        """
        return self._folder_path

    @folder_path.setter
    def folder_path(self, new_value: str) -> None:
        # Input validity
        if not isinstance(new_value, str): raise TypeError(f"The folder_path was expected to be a string but is {type(new_value)}.")
        if not os.path.isdir(new_value): raise ValueError(f"The path {new_value} is not a directory.")
        
        # Set property
        self._folder_path = new_value

    def __len__(self) -> int:
        """The number of instances in the dataset.
        :return: The number of instances in the dataset.
        :rtype: int
        """

        # Get all file names in the data folder
        file_names = os.listdir(self.folder_path)

        # Filter for files that start with 'trainimg-' or 'testimg-'
        instance_count = 0
        for file_name in file_names:
            if 'inputdata' in file_name: instance_count += 1
        
        # Output
        return instance_count
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        file_path = os.path.join(self.folder_path, f"img-{index}-targetdata.txt" )
        
        with open(file=file_path, mode='r') as file_handle:
            text_data = file_handle.read()

        # Parse
        # The first 10 columns one-hot encode the label. All but the first rows can be ignored.
        # The next 4 columns encode the dx, dy, end of stroke (0,1) and end of sequence (0, 1). The end of sequence can be ignored
        lines = text_data.split('\n')
        y = torch.Tensor([(float)(entry) for entry in lines[0].split(' ')[:10]])
        
        x = [None] * len(lines)
        for l, line in enumerate(lines):
            if len(line) > 0: # Exclude possible empty lines
                x[l] = [(float)(entry) for entry in line.split(' ')[10:13]]
        del x[l:]
        
        x = torch.Tensor(x)

        return x,y

if __name__=="__main__":
    path = bpuu.map_path_to_os(path=os.path.join("tests","data","Pen Stroke MNIST"))
    dataset = PenStrokeMNIST(folder_path=path)
    #dataset = Sinusoids(instance_count=10, duration=3.0, sampling_rate=50, frequency_count=2, noise_range=0.02)
    data_loader = DataLoader(dataset, batch_size=3, collate_fn=lambda sequences: collate_function(sequences=sequences, batch_first=True))
    
    for X, y in data_loader:
        print(X.shape, y.shape)