import random, numpy as np
import torch
import json, sys, os
import copy as cp
sys.path.append(os.path.abspath(""))
from src.briann.utilities import file_management as bpufm
from src.briann.network import core as bpnc

# Load Configuration
from experiments.configurations import example_briann as briann_loader
training_configuration = briann_loader.training_configuration

# Extract model
briann = training_configuration["model"]

# Extract data
data_iterator = training_configuration["data_iterator"]
class_count = training_configuration["class_count"]
sample_X, sample_y = next(data_iterator)
print(f"Input data has shape X: {sample_X.shape} and y: {sample_y.shape} with {class_count} classes.")

# Sample pass
briann.load_next_stimulus_batch(X=sample_X)
for i in range(10): briann.step()

y_hat = briann.get_area_at_index(index=5).output_time_frame_accumulator._time_frame.state
print(f"After a trial forward pass, the total energy of the output is {torch.sum(y_hat**2)}")

# Train
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(briann.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(data_iterator):
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Every data instance is an input + label pair
        X_batch, y_batch = data
        
        # Make model load data
        briann.load_next_stimulus_batch(X=X_batch)
        
        # Make predictions for this batch
        for j in range(10):
            briann.step()
        y_hat = briann.get_area_at_index(index=5).output_time_frame_accumulator._time_frame.state

        # Compute the loss and its gradients
        loss = loss_fn(y_hat, y_batch-1) # Subtract 1 from y_batch because the Sinusoids datasets outputs y_batches that start their labels at 1, rather than 0
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

state_dict_path = bpufm.map_path_to_os(path=os.path.join("experiments/training/example briann state_dict.pth"))
torch.save(obj=briann.state_dict(), f=state_dict_path)
briann.load_state_dict(torch.load(state_dict_path))

