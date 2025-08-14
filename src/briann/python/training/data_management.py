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

class SequentialMNIST(Dataset):
    
    def __init__(self, folder_path: str, portion: str = 'both', padding: str = 'pre', **kwargs) -> "SequentialMNIST":
        # Call super
        super().__init__(**kwargs)

        # Map path to operating system
        if os.name == 'nt': # Windows
            folder_path = str(PureWindowsPath(folder_path))
        elif os.name == 'posix':
            folder_path = str(PurePosixPath(folder_path))
        
        # Set attributes
        self._folder_path = folder_path

        # Map portion
        TRAIN_COUNT = 60000
        TEST_COUNT = 10#000
        if portion == 'both': 
            portions = ['train','test']
            n = TRAIN_COUNT + TEST_COUNT
        elif portion == 'train': 
            portions = [portion]
            n = TRAIN_COUNT
        elif portion == 'test':
            portions = [portion]
            n = TEST_COUNT
        else:
            raise ValueError(f"The input called portion should be a string from the set ['train','test','both'], but it is {portion}.")
                        
        # Load portions
        index = 0
        X = [None] * n; y = [None] * n
        max_sequence_length = 0
        if 'train' in portions:
            for i in range(TRAIN_COUNT):
                X[index], y[index] = self.load_input_file(portion='train', index=i)
                max_sequence_length = max(max_sequence_length, X[index].shape[0])
                index += 1
        if 'test' in portions:
            for i in range(TEST_COUNT):
                X[index], y[index] = self.load_input_file(portion='test', index=i)
                max_sequence_length = max(max_sequence_length, X[index].shape[0])
                index += 1

        # Put in standard form
        for i in range(n):
            # Pad sequences
            if padding == 'pre': X[i] = torch.concat([torch.zeros([max_sequence_length - X[i].shape[0], 3]), X[i]], dim=0)
            elif padding == 'post': X[i] = torch.concat([X[i], torch.zeros([max_sequence_length - X[i].shape[0], 3])], dim=0)
            else: raise ValueError(f"The input called padding should be a string from the set ['pre']['post'], but it is {padding}.")

            # Expand axes
            X[i] = X[i][torch.newaxis,:]
        self.X = torch.concat(tensors=X, dim=0); self.y = torch.concat(tensors=y, dim=0)


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def load_input_file(self, portion: str, index: int):

        file_path = os.path.join(self._folder_path, f"{portion}img-{index}-targetdata.txt" )
        
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
    mnist = SequentialMNIST(folder_path=os.path.join("C:\\","Users","P70057764","Downloads","sequences"), portion='test')
    data_loader = DataLoader(mnist, batch_size=2)
    

    for x, y in data_loader:
        print(x.shape, y.shape)