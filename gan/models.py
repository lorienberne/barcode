import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, ngf=32):
        super(Generator, self).__init__()

        # Initial convolution block
        self.conv1 = self._make_layer(
            1, ngf, k_size=9, stride=1, padding=4, activation="lrelu", use_bn=True
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Adding max pooling

        self.conv2 = self._make_layer(
            ngf, ngf * 2, k_size=3, stride=1, padding=1, activation="lrelu", use_bn=True
        )

        self.conv3 = self._make_layer(
            ngf * 2,
            ngf * 4,
            k_size=3,
            stride=1,
            padding=1,
            activation="lrelu",
            use_bn=True,
        )

        # Residual blocks
        self.res1 = self._make_res_block(ngf * 4, ngf * 4)
        self.res2 = self._make_res_block(ngf * 4, ngf * 4)
        self.res3 = self._make_res_block(ngf * 4, ngf * 4)

        # Up-sampling
        self.deconv1 = self._make_deconv_layer(
            ngf * 4, ngf * 2, activation="lrelu", use_bn=True
        )
        self.deconv2 = self._make_deconv_layer(
            ngf * 2, ngf, activation="lrelu", use_bn=True
        )

        # Final layer
        self.conv4 = nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()

    def _make_layer(
        self,
        in_channels,
        out_channels,
        k_size,
        stride,
        padding,
        activation,
        use_bn=False,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=k_size,
                stride=stride,
                padding=padding,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _make_res_block(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        return nn.Sequential(*layers)

    def _make_deconv_layer(self, in_channels, out_channels, activation, use_bn=False):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool(c1)  # Applying max pooling
        c2 = self.conv2(p1)
        p2 = self.pool(c2)  # Applying max pooling
        c3 = self.conv3(p2)

        r1 = self.res1(c3) + c3
        r2 = self.res2(r1) + r1
        r3 = self.res3(r2) + r2

        d1 = self.deconv1(r3)
        d2 = self.deconv2(d1)
        d2 = d2 + c1  # Adjusting the skip connection

        conv4 = self.conv4(d2)
        conv4 = conv4 + x  # Skip connection
        out = self.tanh(conv4)
        return torch.sigmoid(out)  # Adjusting the output


# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x
