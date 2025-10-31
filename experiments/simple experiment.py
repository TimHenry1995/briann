import random, numpy as np
import torch
import json, sys, os
import copy as cp
sys.path.append(os.path.abspath(""))
from src.briann.python.utilities import file_management as bpufm
from src.briann.python.network import core as bpnc

# Load Briann
path = bpufm.map_path_to_os(path="tests/briann 2.json")
with open(path, 'r') as file:
    configuration = json.loads(file.read())

briann = bpnc.BrIANN(configuration=configuration)


training_loader = cp.deepcopy(briann.get_area_at_index(index=0).data_loader) # Area 0 is the source area
class_count = training_loader.dataset.frequency_count

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(briann.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        # Make model load data
        briann.load_next_stimulus_batch()
        
        # Every data instance is an input + label pair
        _, y = data
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        for j in range(10):
            briann.step()
        y_hat = briann.get_area_at_index(index=5).output_time_frame_accumulator._time_frame.state

        # Compute the loss and its gradients
        loss = loss_fn(y_hat, y-1)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch_index * len(training_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Work out reproducibiblity 
# Start training
train_one_epoch(epoch_index=0, tb_writer=None)

state_dict_path = bpufm.map_path_to_os(path="tests/briann 2 state_dict.json")
torch.save(obj=briann.state_dict(), f=state_dict_path)
briann.load_state_dict(torch.load(state_dict_path))
p=0
