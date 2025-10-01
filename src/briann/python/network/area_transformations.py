import torch

class SimpleDenseTransformation(torch.nn.Module):

    def __init__(self, input_dimensionality, output_dimensionality):
        super().__init__()

        self._linear = torch.nn.Linear(input_dimensionality, output_dimensionality)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self._sigmoid(self._linear(x))
    