from torch import nn


class ConvAE(nn.Module):
    def __init__(self, in_kernels, code_kernels, kernel_size, pool, act=nn.ReLU()):
        super().__init__()

        conv = nn.Conv2d(in_kernels, code_kernels, kernel_size)
        avgpool = nn.AvgPool2d(pool)

        self.encoder = nn.Sequential(*[conv, act, avgpool])

        upsample = nn.Upsample(scale_factor=pool)
        transconv = nn.ConvTranspose2d(code_kernels, in_kernels)

        self.decoder = nn.Sequential(*[upsample, act, transconv])

    def forward(self, x):
        code = self.encoder(x)
        reconstruction = self.decoder(code)
        return reconstruction, code
