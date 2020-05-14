from torch import nn


class LinearAE(nn.Module):
    def __init__(self, input_units, code_units):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_units, code_units),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.ReLU(),
                                     nn.Linear(code_units, input_units))

    def forward(self, x):
        code = self.encoder(x)
        reconstruct = self.decoder(code)
        return reconstruct, code
