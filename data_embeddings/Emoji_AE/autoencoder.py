import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from dataset import Emoji
import os
import numpy as np
from typing import List

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

patch_size = 64
data_dir = './data/emoji'
model_path = './conv_autoencoder.pth'


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    x = x.numpy() * 255
    x = x.astype(np.uint8)
    x = np.transpose(x, (0, 2, 3, 1))
    return x


num_epochs = 1000
batch_size = 128
learning_rate = 1e-3
test = True


class AutoEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        hidden_dims.reverse()


    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        z = self.fc(x)

        recon_x = self.decoder_input(z)
        recon_x = recon_x.view(-1, self.hidden_dims[-1], 2, 2)
        recon_x = self.final_layer(self.decoder(recon_x))
        return recon_x, z

hidden_dims = [32, 64, 64, 64, 64]
model = AutoEncoder(3, 300, hidden_dims=hidden_dims).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

t = transforms.Compose([transforms.Resize((patch_size, patch_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = torch.utils.data.DataLoader(
    Emoji(data_dir, transform=t, split='train'),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    Emoji(data_dir, transform=t, split='test'),
    batch_size=batch_size, shuffle=False)


def train():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = Variable(img).cuda()
            recon_x, z = model(img)
            loss = criterion(recon_x, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch+1, num_epochs, total_loss))
        if (epoch + 1) % 50 == 0:
            pic = to_img(recon_x.detach().cpu().data)
            save_image(pic, './dc_img/image_{}.png'.format(epoch + 1))

    torch.save(model.state_dict(), model_path)


def test():
    model.load_state_dict(torch.load(model_path))
    model.eval()
    total_loss = 0
    for data in test_loader:
        img, _ = data
        img = Variable(img).cuda()
        recon_x, z = model(img)
        loss = criterion(recon_x, img)
        total_loss += loss.data

        pic = to_img(recon_x.detach().cpu().data)
        pic = torch.Tensor(pic)
        save_image(pic, './dc_img/test.png')


if __name__ == '__main__':
    if test:
        test()
    else:
        train()
