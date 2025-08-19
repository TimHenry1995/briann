# https://edwin-de-jong.github.io/blog/mnist-sequence-data/
import sys
from pathlib import PureWindowsPath, PurePosixPath
import os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import components as bnc
import random
import torch
from collections import deque
from torch.utils.data import Dataset, DataLoader

class PenStrokeMNIST(Dataset):
    
    def __init__(self, folder_path: str, portion: str = 'both', padding: str = 'pre', **kwargs) -> "PenStrokeMNIST":
        # Call super
        super().__init__(**kwargs)

        # Map path to operating system
        if os.name == 'nt': # Windows
            folder_path = str(PureWindowsPath(folder_path))
        elif os.name == 'posix':
            folder_path = str(PurePosixPath(folder_path))
        
        # Set attributes
        self.folder_path = folder_path
        self.portion = portion
        self.padding = padding
        
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
        if not os.path.exists(new_value): raise ValueError(f"The path {new_value} does not exist.")
        try:
            self.load_input_file(portion='train', index=0, folder_path=new_value)
        except: 
            raise ValueError(f"The path {new_value} does not contain the data in its expected format")
        
        # Set property
        self._folder_path = new_value

    @property
    def portion(self) -> str:
        """The portion of the data loaded by self. Is a string in ['train','test','both'].

        :return: The portion of the data loaded by self.
        :rtype: str
        """
        return self._portion
    
    @portion.setter
    def portion(self, new_value: str) -> None:
        # Check input validity
        if not isinstance(new_value, str): raise ValueError(f"The portion was expected to be a str but is a {type(new_value)}.")
        new_value = new_value.lower()
        if new_value not in ['train','test','both']: raise ValueError(f"The portion should be a string from the set ['train','test','both'], but it is {new_value}.")
            
        # Set property
        self._portion = new_value

        # Map portion
        if new_value == 'both': 
            self._max_sequence_length = 117
            self._instance_count = 70000
        elif self.portion == 'train': 
            self._max_sequence_length = 117
            self._instance_count = 60000
        elif self.portion == 'test':
            self._max_sequence_length = 108
            self._instance_count = 10000

    @property
    def padding(self) -> str:
        """A string indicating whether padding should be applied 'pre' or 'post' the sequence of pen-strokes for each instance.

        :return: A string indicating which padding should be applied.
        :rtype: str
        """
        return self._padding
    
    @padding.setter
    def padding(self, new_value: str) -> None:
        # Check input validity
        if not isinstance(new_value, str): raise ValueError(f"The padding was expected to be a str but is a {type(new_value)}.")
        new_value = new_value.lower()
        if new_value not in ['pre','post']: raise ValueError(f"The padding should be a string from the set ['pre','post'], but it is {new_value}.")
            
        # Set property
        self._padding = new_value

    def __len__(self):
        return self._instance_count
    
    def __getitem__(self, idx):

        # Load
        X, y = self.load_input_file(portion=self.portion, index=idx, folder_path=self.folder_path)
        
        # Put in standard form
        if self.padding == 'pre': X = torch.concat([torch.zeros([self._max_sequence_length - X.shape[0], 3]), X], dim=0)
        elif self.padding == 'post': X = torch.concat([X, torch.zeros([self._max_sequence_length - X.shape[0], 3])], dim=0)
        
        # Output
        return X, y

    def load_input_file(self, portion: str, index: int, folder_path: str = None):

        file_path = os.path.join(folder_path, f"{portion}img-{index}-targetdata.txt" )
        
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
    mnist = PenStrokeMNIST(folder_path=os.path.join("C:\\","Users","P70057764","Downloads","sequences"), portion='train')
    data_loader = DataLoader(mnist, batch_size=2)
    

    for x, y in data_loader:
        print(x.shape, y.shape)
        break