import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.Variable as Variable


class Generator(nn.Module):
    def __init__(self, n_samples, dataset, dim=64, output_dim=784): # CIFAR10 : 3072
        super(MNISTGenerator, self).__init__()
        self.n_samples = n_samples
        self.mode = mode
        self.dim = dim
        self.output_dim = output_dim
        self.dataset = dataset

        self.lin1 = nn.Linear(128, 4 * 4 * 4 * self.dim)
        self.deconv1 = nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 5)
        self.deconv2 = nn.ConvTranspose2d(2 * self.dim, self.dim, 5)
        if(self.dataset == "mnist"):
            self.deconv3 = nn.ConvTranspose2d(self.dim, 1, 5)
        else :
            self.deconv3 = nn.ConvTranspose2d(self.dim, 3, 5)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, noise=None):
        if(noise == None):
            noise = torch.randn(n_samples, 128)

        output = self.lin1(noise)
        output = self.relu(output)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.deconv1(output)
        output = self.relu(output)
        output = output[:, :, :7, :7]
        output = self.deconv2(output)
        output = self.relu(output)
        output = self.deconv3(output)
        output = self.sigmoid(output)

        return output.view(-1, self.output_dim)

class Discriminator(nn.Module):
    def __init__(self, dim=64, dataset):
        self.dim = dim
        self.dataset = dataset
        if(dataset == "mnist"):
            self.conv1 = nn.Conv2d(1, self.dim, 5, stride=2)
        else :
            self.conv1 = nn.Conv2d(3, self.dim, 5, stride=2)
        self.conv2 = nn.Conv2d(self.dim, 2 * self.dim, 5, stride=2)
        self.conv3 = nn.Conv2d(2 * self.dim, 4 * self.dim, 5, stride=2)
        self.lin1 = nn.Linear(4 * 4 * 4 * self.dim, 1)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        if(self.dataset == "mnist"):
            output = inputs.view(-1, 1, 28, 28) # B, Ch, R, C
        else : 
            output = inputs.view(-1, 3, 32, 32)
        output = self.conv1(output)
        output = self.leakyrelu(output)
        output = self.conv2(output)
        output = self.leakyrelu(output)
        output = self.conv3(output)
        output = self.leakyrelu(output)
        output = output.view(-1, 4 * 4 * 4 * self.dim)
        output = self.lin1(output)
        return output.view(-1)
