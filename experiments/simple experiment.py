import random, numpy as np
import torch
import json, sys, os
sys.path.append(os.path.abspath(""))
from src.briann.python.utilities import file_management as bpufm
from src.briann.python.network import core as bpnc

# Load Briann
path = bpufm.map_path_to_os(path="tests/briann 2.json")
with open(path, 'r') as file:
    configuration = json.loads(file.read())

briann = bpnc.BrIANN(configuration=configuration)

briann.load_next_stimulus_batch()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(briann.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Work out reproducibiblity 
# Start training


state_dict_path = bpufm.map_path_to_os(path="tests/briann 2 state_dict.json")
torch.save(obj=briann.state_dict(), f=state_dict_path)
briann.load_state_dict(torch.load(state_dict_path))
