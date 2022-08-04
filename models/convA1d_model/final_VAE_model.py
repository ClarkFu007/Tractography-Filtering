import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC


def count_parameters(model):
    print("The number of trainable parameters is {}."
          .format(sum(p.numel() for p in model.parameters() if p.requires_grad)))


class PrintLayer(nn.Module):
    def __init__(self, position):
        super(PrintLayer, self).__init__()
        self.position = position

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.position, x.shape)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, X):
        return X.view(self.shape)


class AutoencoderModel(nn.Module, ABC):
    """
        Variational Autoencoder with 1-D convolutional layers.
    """
    def __init__(self, dim_num, latent_dim_num):
        """
        :param dim_num: number of dimensions for each point
        :param latent_dim_num: number of dimensions of latent space
        """
        super(AutoencoderModel, self).__init__()
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=dim_num, out_channels=32, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            #PrintLayer('1'),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            #PrintLayer('2'),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(),
            #PrintLayer('3'),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(),
            #PrintLayer('4'),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3,
                      stride=2, padding=1),
            nn.BatchNorm1d(512), nn.LeakyReLU(),
            #PrintLayer('5'),
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3,
                      stride=1, padding=1),
            #PrintLayer('6'),
            Reshape(-1, 1024 * 32),
        )

        # Latent space
        self.z_mean = nn.Linear(1024 * 32, latent_dim_num)
        self.z_log_var = nn.Linear(1024 * 32, latent_dim_num)

        # Decoder
        self.decoder = nn.Sequential(
            #PrintLayer('7'),
            nn.Linear(latent_dim_num, 1024 * 32),
            Reshape(-1, 1024, 32),
            #PrintLayer('8'),
            nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=3,
                               stride=1, padding=1),
            nn.BatchNorm1d(512), nn.LeakyReLU(),
            #PrintLayer('9'),
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(),
            #PrintLayer('10'),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128), nn.LeakyReLU(),
            #PrintLayer('11'),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3,
                               stride=1, padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            #PrintLayer('12'),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3,
                               stride=1, padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            #PrintLayer('13'),
            nn.ConvTranspose1d(in_channels=32, out_channels=dim_num, kernel_size=3,
                               stride=1, padding=1),
            #PrintLayer('14'),
        )

    def _reparameterize(self, z_mu, z_log_var):
        if torch.cuda.is_available():  # 0.1 is good
            eps = torch.normal(0, 0.5, size=(z_mu.size(0), z_mu.size(1))).to(z_mu.get_device())
        else:
            eps = torch.normal(0, 0.5, size=(z_mu.size(0), z_mu.size(1)))
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        """
        mu = torch.mean(z_mu)
        std = torch.mean(torch.sqrt(torch.exp(z_log_var / 2.)))
        if torch.cuda.is_available():
            z = torch.normal(float(mu), float(std), size=(z_mu.size(0), z_mu.size(1))).to(z_mu.get_device())
        else:
            z = torch.normal(float(mu), float(std), size=(z_mu.size(0), z_mu.size(1)))
        """

        return z

    def _encoding_fn(self, X):
        """
           kernel_size=3, stride=1, padding=0
           X: torch.Size([25, 3, 128])
        """
        x = self.encoder(X)
        x = x.view(x.size(0), -1)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)

        z_mean = 0.5 * torch.sigmoid(z_mean)   # Good!
        #z_mean = 0.2 * torch.sigmoid(z_mean) + 0.3
        z_log_var = - F.relu(z_log_var)
        encoded_x = self._reparameterize(z_mean, z_log_var)
        return encoded_x, z_mean, z_log_var

    def forward(self, X, encode, reconstruct):
        if reconstruct:
            reconstructed_data = self.decoder(X)
            return reconstructed_data

        else:
            encoded_data, z_mean, z_log_var = self._encoding_fn(X=X)
            if encode:
                return encoded_data, z_mean, z_log_var
            else:
                decoded_data = self.decoder(encoded_data)
                return decoded_data, z_mean, z_log_var


if __name__ == '__main__':
    autoencoder_model = AutoencoderModel(dim_num=3, latent_dim_num=32)
    input_data = torch.randn((64, 3, 128))
    print("input_data.shape", input_data.shape)
    output_data, _, _ = autoencoder_model(input_data, encode=False, reconstruct=False)
    print("output_data.shape", output_data.shape)
    assert input_data.shape == output_data.shape
    count_parameters(autoencoder_model)