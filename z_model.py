import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ZModel(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.optimizer = None

    def print_network(self):
        """Print out the network information."""
        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print(self)
        print(self.name)
        print("The number of parameters: {}".format(num_params))


class ZEncoder(ZModel):
    """Generator network."""

    def __init__(self, name, conv_dim=64, c_dim=5, repeat_num=6, create_optimizer=None):
        super(ZEncoder, self).__init__(name)

        layers = [nn.Conv2d(3 + c_dim, conv_dim, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False),
                  nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(4):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        self.out_dim = curr_dim

        self.main = nn.Sequential(*layers)
        if create_optimizer is not None:
            self.optimizer = create_optimizer(self)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


class ZDecoder(ZModel):
    """Generator network."""

    def __init__(self, name, conv_dim, create_optimizer=None):
        super(ZDecoder, self).__init__(name)

        layers = []
        curr_dim = conv_dim

        # Up-sampling layers.
        for i in range(4):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=(4, 4),
                                             stride=(2, 2), padding=(1, 1), bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)
        if create_optimizer is not None:
            self.optimizer = create_optimizer(self)

    def forward(self, x):
        return self.main(x)


class ZDiscriminator(ZModel):
    """Discriminator network with PatchGAN."""

    def __init__(self, name, conv_dim, c_dim, create_optimizer=None):
        super(ZDiscriminator, self).__init__(name)
        layers = [nn.Linear(conv_dim, c_dim), nn.LeakyReLU(0.01)]

        self.main = nn.Sequential(*layers)

        if create_optimizer is not None:
            self.optimizer = create_optimizer(self)

    def forward(self, x):
        h = x.mean([2, 3])
        return self.main(h)
