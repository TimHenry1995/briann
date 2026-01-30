import sys
import os
sys.path.append(os.path.abspath(""))
from briann.utilities import random as bpur
from typing import Tuple
import torch
from torch.utils.data import Dataset
from typing import List

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
    """A dataset of sinusoids with varying frequencies. Each instance is a time-series of a sinusoid with one frequency and added noise.
    The frequency is determined by the index of the instance. The dataset can return the input data, the labels, or both.
    
    :param instance_count: Sets the :py:meth:`~.Sinusoids.instance_count` property.
    :type instance_count: int
    :param duration: Sets the :py:meth:`~.Sinusoids.duration` property.
    :type duration: float
    :param sampling_rate: Sets the :py:meth:`~.Sinusoids.sampling_rate` property.
    :type sampling_rate: float
    :param frequency_count: Sets the :py:meth:`~.Sinusoids.frequency_count` property.
    :type frequency_count: int
    :param noise_range: Sets the :py:meth:`~.Sinusoids.noise_range` property.
    :type noise_range: float
    :param return_X: Sets the :py:meth:`~.Sinusoids.return_X` property.
    :type return_X: bool
    :param return_y: Sets the :py:meth:`~.Sinusoids.return_y` property.
    :type return_y: bool

    :return: The sinusoid dataset.
    :rtype: Sinusoids
    """

    def __init__(self, instance_count: int, 
                 duration: float, 
                 sampling_rate: float, 
                 frequency_count: int, 
                 noise_range: float, 
                 return_X: bool = True,
                 return_y: bool = True,
                 ) -> None:
        # Call super
        super(Sinusoids, self).__init__()

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

        if not isinstance(return_X, bool): raise TypeError(f"The return_X was expected to be a boolean but is {type(new_value)}.")
        
        if not isinstance(return_y, bool): raise TypeError(f"The return_y was expected to be a boolean but is {type(new_value)}.")
        
        # Set properties
        self._instance_count = instance_count
        self._duration = (float)(duration)
        self._sampling_rate = sampling_rate
        self._frequency_count = frequency_count
        self._noise_range = noise_range
        self._return_X = return_X
        self._return_y = return_y

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

    @property
    def return_X(self) -> bool:
        return self._return_X
    
    @property
    def return_y(self) -> bool:
        return self._return_y
    
    def __len__(self):
        return self.instance_count
    
    def __getitem__(self, idx):
        # Choose frequency
        frequency = idx % self._frequency_count + 1
        y = frequency

        # Generate sinusoid
        X = torch.zeros([(int)(self._sampling_rate * self._duration), 1])
        ts = torch.linspace(start=0.0, end=self._duration, steps=(int)(self._sampling_rate * self._duration))
        X[:,0] = torch.sin(ts * 2 * torch.pi * frequency)
        noise = torch.Tensor(bpur.StatelessRandom.pseudo_uniform(lower_bound=-self._noise_range/2, upper_bound=self._noise_range/2, seed=idx, count=X.shape[0]))
        X[:,0] += noise

        # Output
        if self.return_X and not self.return_y:
            return X
        if not self.return_X and self.return_y:
            return y
        if self.return_X and self.return_y:
            return X, y

