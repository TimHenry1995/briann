import torch.nn as nn
import torch

import sys, os
sys.path.append(os.path.abspath(""))
from experiments.vggish import model_loader as model_loader


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims= dims

    def forward(self, x):
        return torch.transpose(x, *self.dims)


def make_regular_vggish(state_dict_path):
    
    # Add layers
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    layers += [
        Transpose([1,3]),
        Transpose([1,2]),
        nn.Flatten(),
        nn.Linear(512 * 4 * 6, 4096),
        nn.ReLU(True),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Linear(4096, 128),
    ]

    vggish = torch.nn.Sequential(*layers)
    vggish.load_state_dict(torch.load(state_dict_path))
    return vggish

def load_vggish_weights_into_briann(briann, state_dict_path):
    vggish = make_regular_vggish(state_dict_path=state_dict_path)

    # Map
    briann_weight_indices = [1,3,5,6,8,9,11,12,13]
    vggish_weight_indices = [0,3,6,8,11,13,19,21,23]
    for i,j in zip(briann_weight_indices, vggish_weight_indices):
        briann_area = briann.get_area_at_index(index=i) 
        vggish_layer = vggish[j]
        
        if i != 11:
            briann_area.state_dict()['_transformation.0.weight'].data = vggish_layer.state_dict()['weight'].data.clone()
            briann_area.state_dict()['_transformation.0.bias'].data = vggish_layer.state_dict()['weight'].data.clone()
        else:
            briann_area.state_dict()['_transformation.3.weight'].data = vggish_layer.state_dict()['weight'].data.clone()
            briann_area.state_dict()['_transformation.3.bias'].data = vggish_layer.state_dict()['weight'].data.clone()

