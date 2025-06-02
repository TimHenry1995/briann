# https://edwin-de-jong.github.io/blog/mnist-sequence-data/
import sys
import os
sys.path.append(os.path.abspath(""))
from src.briann.python.network import components as bnc
import random
import torch
from collections import deque
class MNIST_Sequence():

    TRAIN_COUNT = 60000
    TEST_COUNT = 10000

    def __init__(self, batch_size: int, folder_path: str, from_train: bool, seed: int):
        self._batch_size = batch_size
        self._folder_path = folder_path
        self._from_train = from_train
        random.seed(seed)
        self._indices = list(range(0, self.TRAIN_COUNT if from_train else self.TEST_COUNT)); random.shuffle(self._indices)
        self._current_batch_start_index = 0

    def load_input_file(self, index: int):

        set_name = "train" if self._from_train else "test"
        file_path = os.path.join(self._folder_path, f"{set_name}img-{index}-targetdata.txt" )
        
        with open(file=file_path, mode='r') as file_handle:
            text_data = file_handle.read()

        # Parse
        # The first 10 columns one-hot encode the label. All but the first row can be ignored.
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
            

    def batch_generator(self):
        
        while self._current_batch_start_index + self._batch_size < len(self._indices):
            indices = self._indices[self._current_batch_start_index: self._current_batch_start_index + self._batch_size]
            self._current_batch_start_index += self._batch_size

            X = [None] * self._batch_size
            Y = [None] * self._batch_size
            max_sequence_length = 0

            # Load batch
            for i, index in enumerate(indices):
                X[i], Y[i] = self.load_input_file(index=index)
                max_sequence_length = max(max_sequence_length, X[i].shape[0])

            # Put X in standard form
            for i in range(len(indices)):
                # Pad sequences
                X[i] = torch.concat([torch.zeros([max_sequence_length - X[i].shape[0], 3]), X[i]], dim=0)

                # Expand axes
                X[i] = X[i][torch.newaxis,:]

            # Concatenate the different instances
            X = torch.concat(X, dim=0) # Shape == [batch size, max sequence length, 3]

            # Convert X to TimeFrame objects
            time_frames = deque([])
            for t in range(max_sequence_length):
                time_frames.appendleft(bnc.TimeFrame(state=X[:,t,:], index=t, start_time=(float)(t), duration=1.0))

            yield time_frames, Y


if __name__=="__main__":
    mnist = MNIST_Sequence(batch_size = 4, folder_path=os.path.join("C:\\","Users","P70057764","Downloads","sequences"), from_train=True, seed=42)
    batch_generator = mnist.batch_generator()
    time_frames, Y = next(batch_generator)

    print(time_frames[0].state)