import sys, os
sys.path.append(os.path.abspath(""))
from experiments.sinusoids.data_loader import Sinusoids
import vggish_input
import torch
from torch.utils.data import Dataset
from typing import Tuple
from briann.utilities import random as bpur
from typing import List

class VGGishSpectrograms(Dataset):
    """A dataset of spectrogram slices as expected by VGGish, each instance is a spectrogram taken from the waveforms of the given `waveforms` dataset. 
    The `waveform` is treated as sampled at 16 kHz.

    :param waveforms: A dataset of waveforms to convert to spectrograms.
    :type waveforms: Dataset
    """

    def __init__(self, waveforms: Dataset
                 ) -> None:
        # Call super
        super(VGGishSpectrograms, self).__init__()

        # Check input validity
        if not isinstance(waveforms, Sinusoids): raise TypeError(f"The sinusoids argument was expected to be a Sinusoids Dataset instance but is of type {type(waveforms)}.")
        self._waveforms_ = waveforms
        
    @property
    def return_X(self) -> bool:
        return self._waveforms_._return_X
    
    @property
    def return_y(self) -> bool:
        return self._waveforms_._return_y
    
    def __len__(self):
        return len(self._waveforms_)
    
    def __getitem__(self, idx):
        # Draw sinusoid from dataset
        if self.return_X and self.return_y:
            waveform, y = self._waveforms_[idx]
            X = vggish_input.waveform_to_examples(waveform.numpy())
            X = torch.Tensor(X)
            return X, y
        elif self.return_X and not self.return_y:
            waveform = self._waveforms_[idx]
            X = vggish_input.waveform_to_examples(waveform.numpy(), sample_rate=16000)
            X = torch.Tensor(X)
            return X
        elif not self.return_X and self.return_y:
            y = self._waveforms_[idx]
            return y