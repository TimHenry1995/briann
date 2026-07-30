import torch

class SimpleDenseTransformation(torch.nn.Module):

    def __init__(self, input_dimensionality, output_dimensionality):
        super().__init__()

        self._linear = torch.nn.Linear(input_dimensionality, output_dimensionality, bias=True)
        #self._activation = torch.nn.ReLU()

    def forward(self, x):
        return self._linear(x)#self._activation(self._linear(x))
    