import functools

from deepul.hw4_helper import *
import deepul.pytorch_util as ptu
import warnings

warnings.filterwarnings('ignore')
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from tqdm import trange, tqdm_notebook
import torchvision
import pickle


def visualize_q1():
    visualize_q1_dataset()


def load_q1_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n // 2,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(n // 2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1


def load_q2_data(n=20000):
    assert n % 2 == 0
    gaussian1 = np.random.normal(loc=-1, scale=0.03, size=(n // 2,))
    gaussian2 = np.random.normal(loc=1, scale=0.03, size=(n // 2,))
    data = (np.concatenate([gaussian1, gaussian2]) + 1).reshape([-1, 1])
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1


def load_warmup_data(n=20000):
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n // 2,))
    data = gaussian1.reshape([-1, 1]) + 1
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    return 2 * scaled_data - 1


def train(model, train_loader, k, batch_size):
    model.train()
    train_losses = []
    for x, i in zip(train_loader, range(len(train_loader.dataset))):
        loss = model.train_d(x)
        train_losses.append(loss)
        if i % k == 0:
            model.train_g(batch_size)
        # print(f"iteration {i} loss {loss}")
    return train_losses


def train_epochs(model, train_loader, train_args, tmp_model=None):
    epochs, lr, k, bs = train_args['epochs'], train_args['lr'], train_args["k"], train_args["bs"]
    train_losses = []
    for epoch in range(epochs):
        train_loss = train(model, train_loader, k, bs)
        train_losses.extend(train_loss)
        print(f"epoch {epoch} loss {sum(train_loss) / len(train_loss)}")
        if epoch == 0 and not (tmp_model is None):
            tmp_model.d.load_state_dict(model.d.state_dict())
            tmp_model.g.load_state_dict(model.g.state_dict())
    return train_losses


class Discriminator(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.model = []
        for i in range(len(hidden_units) - 1):
            self.model.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.model.append(nn.LeakyReLU())
        self.model.pop()
        self.model.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)


class Generator(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.model = []
        for i in range(len(hidden_units) - 1):
            self.model.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            self.model.append(nn.LeakyReLU())
        self.model.pop()
        self.model = nn.Sequential(*self.model)

    def forward(self, inputs):
        return self.model(inputs)


class GAN:
    def __init__(self, d_hidden, g_hidden, lr, batch_size, device, non_saturating=False):
        self.d = Discriminator(d_hidden).to(device)
        self.g = Generator(g_hidden).to(device)
        self.batch_size = batch_size
        self.device = device
        self.non_saturating = non_saturating
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=lr, betas=(0, 0.9))
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=lr, betas=(0, 0.9))

    def train_d(self, x):
        x = x.float().to(self.device)
        z = torch.normal(0, 1, size=x.shape).to(self.device)
        gz = self.g(z)
        dx = self.d(x)
        dgz = self.d(gz)
        outs = -(torch.mean(torch.log(dx) + torch.log(1 - dgz)))
        self.d_optimizer.zero_grad()
        outs.backward()
        self.d_optimizer.step()
        return outs.cpu().item()

    def train_g(self, batch_size):
        z = torch.normal(0, 1, size=(batch_size, 1)).to(self.device)
        gz = self.g(z)
        dgz = self.d(gz)
        if self.non_saturating:
            outs = torch.mean(torch.log(1 - dgz))
        else:
            outs = -torch.mean(torch.log(dgz))
        self.g_optimizer.zero_grad()
        outs.backward()
        self.g_optimizer.step()
        return outs.cpu().item()

    def sample(self, size):
        z = torch.normal(0, 1, size=(size, 1)).to(self.device)
        with torch.no_grad():
            gz = self.g(z)
        return gz.cpu().numpy()

    def score(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        with torch.no_grad():
            dx = self.d(x)
        return dx.cpu().numpy()

    def train(self):
        self.d.train()
        self.g.train()

    def eval(self):
        self.d.eval()
        self.g.eval()


def q1_a(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at the end of training
    """

    d_units = [1, 24, 64, 24, 1]
    g_units = [1, 24, 64, 24, 1]
    lr = 5e-4
    batch_size = 128
    k = 4
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = GAN(d_units, g_units, lr, batch_size, device)
    tmp_model = GAN(d_units, g_units, lr, batch_size, device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    train_losses = train_epochs(model, train_loader, dict(epochs=75, lr=5e-4, k=k, bs=batch_size), tmp_model)
    lin_array = np.linspace(-1, 1, 1000)
    return train_losses, tmp_model.sample(5000).flatten(), lin_array, tmp_model.score(
        lin_array.reshape(-1, 1)).flatten(), model.sample(
        5000).flatten(), lin_array, model.score(lin_array.reshape(-1, 1)).flatten()


def q1a():
    q1_save_results('a', q1_a)


def q1_b(train_data):
    """
    train_data: An (20000, 1) numpy array of floats in [-1, 1]

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of samples drawn from your model at epoch #1
    - a numpy array of size (100,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (100,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of samples drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at the end of training
    """
    d_units = [1, 32, 64, 32, 1]
    g_units = [1, 32, 64, 32, 1]
    lr = 5e-4
    batch_size = 128
    k = 4
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = GAN(d_units, g_units, lr, batch_size, device, True)
    tmp_model = GAN(d_units, g_units, lr, batch_size, device, True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    train_losses = train_epochs(model, train_loader, dict(epochs=75, lr=5e-4, k=k, bs=batch_size), tmp_model)
    lin_array = np.linspace(-1, 1, 1000)
    return train_losses, tmp_model.sample(5000).flatten(), lin_array, tmp_model.score(
        lin_array.reshape(-1, 1)).flatten(), model.sample(
        5000).flatten(), lin_array, model.score(lin_array.reshape(-1, 1)).flatten()


def q1b():
    q1_save_results('b', q1_b)


def visualize_q2():
    visualize_q2_data()


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, inputs):
        output = inputs.permute(0, 2, 3, 1).contiguous()
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth).contiguous()
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, inputs):
        output = inputs.permute(0, 2, 3, 1).contiguous()
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.model = nn.Sequential(DepthToSpace(block_size=2),
                                   nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding))

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.model(x)
        return x


class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.down_sample = SpaceToDepth(2)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.down_sample(x)
        x = sum(x.chunk(4, dim=1)) / 4.0
        return self.conv(x)


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1))
        self.up_sample = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        outs = self.model(x)
        shortcuts = self.up_sample(x)
        return outs + shortcuts


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.model = nn.Sequential(nn.ReLU(),
                                   nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
                                   nn.ReLU(),
                                   Downsample_Conv2d(n_filters, n_filters, kernel_size, padding=1))
        self.down_sample = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        outs = self.model(x)
        shortcuts = self.down_sample(x)
        return outs + shortcuts


class ImageGenerator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.linear = nn.Linear(128, 4 * 4 * 256)
        self.model_after_reshape = nn.Sequential(ResnetBlockUp(in_dim=256, n_filters=n_filters),
                                                 ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
                                                 ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
                                                 nn.BatchNorm2d(n_filters),
                                                 nn.ReLU(),
                                                 nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
                                                 nn.Tanh())

    def forward(self, z):
        batch_size = z.shape[0]
        outs = self.linear(z)
        outs = outs.reshape(shape=(batch_size, 256, 4, 4))
        outs = self.model_after_reshape(outs)
        return outs


class ImageDiscriminator(nn.Module):
    def __init__(self, n_filters=128):
        super().__init__()
        self.linear = nn.Linear(128, 1)
        self.model_before_reshape = nn.Sequential(nn.Conv2d(3, n_filters, kernel_size=(3, 3), padding=1),
                                                  nn.ReLU(),
                                                  ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
                                                  ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
                                                  ResnetBlockDown(in_dim=n_filters, n_filters=128))

    def forward(self, x):
        outs = self.model_before_reshape(x)
        outs = torch.mean(outs, dim=[2, 3])
        return self.linear(outs)


class WGANGP:
    def __init__(self, latent_dim, lr, batch_size, lmbda, device):
        self.d = ImageDiscriminator().to(device)
        self.g = ImageGenerator().to(device)
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.lmbda = lmbda
        self.device = device
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=lr, betas=(0, 0.9))
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=lr, betas=(0, 0.9))

    def train_d(self, x):
        x = x.float().to(self.device)
        x = 2 * x - 1
        batch_size = x.shape[0]
        z = torch.normal(0, 1, size=(batch_size, self.latent_dim)).to(self.device)
        gz = self.g(z).detach()
        dx = self.d(x)
        dgz = self.d(gz)
        outs = torch.mean(dx - dgz)
        gradient_penalty = self.cal_gradient_penalty(x, gz)
        outs = self.lmbda * gradient_penalty - outs
        self.d_optimizer.zero_grad()
        outs.backward()
        self.d_optimizer.step()
        return outs.cpu().item()

    def train_g(self, batch_size):
        self.freeze(self.d)
        z = torch.normal(0, 1, size=(batch_size, self.latent_dim)).to(self.device)
        gz = self.g(z)
        dgz = self.d(gz)
        outs = -torch.mean(dgz)
        self.g_optimizer.zero_grad()
        outs.backward()
        self.g_optimizer.step()
        self.reset(self.d)
        return outs.cpu().item()

    def cal_gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.shape[0]
        alpha = torch.rand(size=real_data.shape).to(self.device)
        interpolations = torch.tensor(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)
        outs = self.d(interpolations)
        gradients = torch.autograd.grad(outputs=outs,
                                        inputs=interpolations,
                                        grad_outputs=torch.ones(outs.size()).to(self.device),
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = torch.mean((torch.norm(gradients, dim=1) - 1) ** 2)
        return gradient_penalty

    def sample(self, size):
        self.g.eval()
        z = torch.normal(0, 1, size=(size, self.latent_dim)).to(self.device)
        with torch.no_grad():
            gz = self.g(z)
        return (gz.permute(0, 2, 3, 1).cpu().numpy() + 1) / 2

    def score(self, x):
        x = torch.from_numpy(x).to(self.device).float()
        with torch.no_grad():
            dx = self.d(x)
        return dx.cpu().numpy()

    def freeze(self, sub_model):
        for param in sub_model.parameters():
            param.requires_grad = False

    def reset(self, sub_model):
        for param in sub_model.parameters():
            param.requires_grad = True

    def train(self):
        self.d.train()
        self.g.train()

    def eval(self):
        self.d.eval()
        self.g.eval()


def q2(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1]. 
        The first 100 will be displayed, and the rest will be used to calculate the Inception score. 
    """

    lr = 2e-4
    batch_size = 256
    latent_dim = 128
    k = 5
    lmbda = 10
    # epochs = int(25000 / (train_data.shape[0] / batch_size))
    epochs = 120
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = WGANGP(latent_dim, lr, batch_size, lmbda, device)
    tmp_model = WGANGP(latent_dim, lr, batch_size, lmbda, device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    train_losses = train_epochs(model, train_loader, dict(epochs=epochs, lr=lr, k=k, bs=batch_size), tmp_model)
    samples = model.sample(1000)
    return train_losses, samples


def q2_save():
    q2_save_results(q2)


def visualize_q3():
    visualize_q3_data()


class BiDiscriminator(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outs_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(inputs_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, outs_dim),
                                   nn.Sigmoid())

    def forward(self, inputs):
        return self.model(inputs)


class BiGenerator(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outs_dim):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.model = nn.Sequential(nn.Linear(inputs_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, outs_dim),
                                   nn.Tanh())
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        return self.model(inputs)

    def sample(self, batch_size):
        x = torch.normal(0, 1, size=(batch_size, self.inputs_dim)).to(self.device)
        x = self.model(x)
        x = (x + 1) / 2
        return x.reshape(batch_size, 28, 28).unsqueeze(-1).cpu().numpy()


class BiEncoder(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outs_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(inputs_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.BatchNorm1d(hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, outs_dim))

    def forward(self, inputs):
        return self.model(inputs)


class BiGAN:
    def __init__(self, real_data_dim, latent_dim, lr, device):
        self.device = device
        self.latent_dim = latent_dim
        self.g = BiGenerator(latent_dim, 1024, real_data_dim).to(device)
        self.e = BiEncoder(real_data_dim, 1024, latent_dim).to(device)
        self.d = BiDiscriminator(real_data_dim + latent_dim, 1024, 1).to(device)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=2.5 * 1e-5)
        self.g_e_optimizer = torch.optim.Adam(list(self.g.parameters()) + list(self.e.parameters()), lr=lr,
                                              betas=(0.5, 0.999), weight_decay=2.5 * 1e-5)

    def train_d(self, x):
        x = x.reshape(x.shape[0], -1).float().to(self.device)
        z = torch.normal(0, 1, size=(x.shape[0], self.latent_dim)).to(self.device)
        gz = self.g(z).detach()
        ex = self.e(x).detach()
        z_gz = torch.cat((z, gz), dim=-1)
        ex_x = torch.cat((ex, x), dim=-1)
        d_z_gz = self.d(z_gz)
        d_ex_x = self.d(ex_x)
        loss = -torch.mean(torch.log(d_ex_x) + torch.log(1 - d_z_gz))
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
        return loss.item()

    def train_g_e(self, x):
        self.freeze(self.d)
        x = x.reshape(x.shape[0], -1).float().to(self.device)
        z = torch.normal(0, 1, size=(x.shape[0], self.latent_dim)).to(self.device)
        gz = self.g(z)
        ex = self.e(x)
        z_gz = torch.cat((z, gz), dim=-1)
        ex_x = torch.cat((ex, x), dim=-1)
        d_z_gz = self.d(z_gz)
        d_ex_x = self.d(ex_x)
        loss = torch.mean(torch.log(d_ex_x) + torch.log(1 - d_z_gz))
        self.g_e_optimizer.zero_grad()
        loss.backward()
        self.g_e_optimizer.step()
        self.reset(self.d)
        return loss.item()

    def sample(self, batch_size):
        self.g.eval()
        with torch.no_grad():
            outs = self.g.sample(batch_size)
        return outs

    def reconstruction(self, x):
        self.g.eval()
        self.e.eval()
        x = x.float().to(self.device).reshape(x.shape[0], -1)
        with torch.no_grad():
            e_x = self.e(x)
            g_e_x = self.g(e_x)
        x = (x + 1) / 2
        g_e_x = (g_e_x + 1) / 2
        image_list = []
        for i in range(x.shape[0]):
            image_list.append(x[i].reshape(28, 28).unsqueeze(-1))
        for i in range(x.shape[0]):
            image_list.append(g_e_x[i].reshape(28, 28).unsqueeze(-1))
        results = torch.stack(image_list, dim=0)
        return results.cpu().numpy()

    def freeze(self, sub_model):
        for param in sub_model.parameters():
            param.requires_grad = False

    def reset(self, sub_model):
        for param in sub_model.parameters():
            param.requires_grad = True

    def train(self):
        self.d.train()
        self.g.train()

    def eval(self):
        self.d.eval()
        self.g.eval()


class LinearClassifier(nn.Module):
    def __init__(self, feature_num, label_num):
        super().__init__()
        self.linear = nn.Linear(feature_num, label_num)
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, encoder):
        encoder.eval()
        e_x = encoder(x).detach()
        outs = self.linear(e_x)
        return outs

    def loss(self, x, labels, encoder):
        x = x.reshape(x.shape[0], -1).float().to(self.device)
        labels = labels.to(self.device)
        outs = self(x, encoder)
        return self.criteria(outs, labels)

    def predict(self, x, encoder):
        x = x.reshape(x.shape[0], -1).float().to(self.device)
        outs = self(x, encoder)
        return torch.argmax(outs, dim=1)


def train_classifier(classifier, encoder, epoch, train_loader, test_loader):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    train_losses = []
    eval_losses = []
    for i in range(epoch):
        for x, label in train_loader:
            loss = classifier.loss(x, label, encoder)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.cpu().item())
        eval_losses.append(eval_classifier(classifier, encoder, test_loader))
        print(f"classifier {i}th epoch test loss {eval_losses[-1]}")
    return train_losses, eval_losses


def eval_classifier(classifier, encoder, test_loader):
    eval_loss = 0
    for x, label in test_loader:
        with torch.no_grad():
            loss = classifier.loss(x, label, encoder)
        eval_loss += loss.cpu().item() * x.shape[0]
    return eval_loss / len(test_loader.dataset)


def eval_acc(classifier, encoder, test_loader):
    true_num = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for x, label in test_loader:
        with torch.no_grad():
            pred = classifier.predict(x, encoder)
            true_num += torch.sum(pred.squeeze() == label.to(device).squeeze()).cpu().item()
    return true_num / len(test_loader.dataset)


def q3(train_data, test_data):
    """
    train_data: A PyTorch dataset that contains (n_train, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST, and it may be easiest to directly create a DataLoader from this variable
    test_data: A PyTorch dataset that contains (n_test, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST

    Returns
    - a (# of training iterations,) numpy array of BiGAN minimax losses evaluated every minibatch
    - a (100, 28, 28, 1) numpy array of BiGAN samples that lie in [0, 1]
    - a (40, 28, 28, 1) numpy array of 20 real image / reconstruction pairs
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on the BiGAN encoder evaluated every epoch 
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on a random encoder evaluated every epoch 
    """
    lr = 2e-4
    batch_size = 128
    latent_dim = 50
    k = 1
    epochs = 50
    feature_num = 28 * 28
    label_num = 10
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    bi_gan = BiGAN(feature_num, latent_dim, lr, device)
    train_bi_gan_losses = []
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    for i in range(epochs):
        j = 0
        for x, _ in train_data_loader:
            d_loss = bi_gan.train_d(x)
            train_bi_gan_losses.append(d_loss)
            if j % k == 0:
                g_e_loss = bi_gan.train_g_e(x)
            j = j + 1
        print(f"{i}th epoch BiGAN D loss {d_loss} BiGAN G loss {g_e_loss}")

    samples = bi_gan.sample(100)
    reconstruction = bi_gan.reconstruction(
        torch.from_numpy(2 * (train_data.train_data.numpy()[0:20] / 255) - 1).unsqueeze(1))

    classifier_epoch = 30
    trained_encoder = bi_gan.e
    trained_classifier = LinearClassifier(latent_dim, label_num).to(device)
    random_encoder = BiGAN(feature_num, latent_dim, lr, device).e
    random_classifier = LinearClassifier(latent_dim, label_num).to(device)

    _, trained_eval_losses = train_classifier(trained_classifier, trained_encoder, classifier_epoch,
                                              train_data_loader, test_data_loader)
    _, random_eval_losses = train_classifier(random_classifier, random_encoder, classifier_epoch,
                                             train_data_loader, test_data_loader)
    trained_acc = eval_acc(trained_classifier, trained_encoder, test_data_loader)
    random_acc = eval_acc(random_classifier, random_encoder, test_data_loader)
    print(f"trained encoder {trained_acc} random encoder {random_acc}")
    return train_bi_gan_losses, samples, reconstruction, trained_eval_losses, random_eval_losses


def q3_save():
    q3_save_results(q3)


def visualize_q4():
    visualize_cyclegan_datasets()


# class CycleGenerator(nn.Module):
#     def __init__(self, in_depth, out_depth, n_filter):
#         super().__init__()
#         self.in_depth = in_depth
#         self.out_depth = out_depth
#         self.n_filter = n_filter
#         self.hidden_units = 1024
#         self.latent_dim = 64
#         self.model = nn.Sequential(nn.Linear(self.in_depth * 28 * 28, self.latent_dim),
#                                    nn.ReLU(),
#                                    nn.Linear(self.latent_dim, self.hidden_units),
#                                    nn.ReLU(),
#                                    nn.Linear(self.hidden_units, 2 * self.hidden_units),
#                                    nn.ReLU(),
#                                    nn.Linear(2 * self.hidden_units, 2 * self.hidden_units),
#                                    nn.Linear(2 * self.hidden_units, self.out_depth * 28 * 28),
#                                    nn.Tanh())
#         # self.model = [nn.Conv2d(in_depth, self.n_filter, 5, 1, 2), nn.ReLU()]
#         # self.model.extend([nn.Conv2d(self.n_filter, self.n_filter, 5, 1, 2), nn.ReLU()] * 6)
#         # self.model.extend([nn.Conv2d(self.n_filter, out_depth, 5, 1, 2), nn.Tanh()])
#         # for i in range(2):
#         #     self.model.append(nn.Conv2d(self.n_filter * (2 ** i),
#         #                                 self.n_filter * (2 ** (i + 1)), 3, 2, 1))
#         #     self.model.append(nn.ReLU())
#         # self.model.extend([nn.Conv2d(4 * self.n_filter, 4 * self.n_filter, 3, 1, 1), nn.ReLU()] * 2)
#         # for i in range(2):
#         #     self.model.append(nn.ConvTranspose2d(self.n_filter * 4 // (2 ** i),
#         #                                          self.n_filter * 4 // (2 ** (i + 1)),
#         #                                          kernel_size=3,
#         #                                          stride=2,
#         #                                          padding=1,
#         #                                          output_padding=1))
#         #     self.model.append(nn.ReLU())
#         # self.model.append(nn.Conv2d(self.n_filter, out_depth, 3, 1, 1))
#         # self.model.append(nn.Tanh())
#         # self.model = nn.Sequential(*self.model)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.reshape(batch_size, -1)
#         outs = (self.model(x) + 1) / 2
#         outs = outs.reshape(batch_size, self.out_depth, 28, 28)
#         return outs


# class CycleDiscriminator(nn.Module):
#     def __init__(self, in_depth, n_filter):
#         super().__init__()
#         self.n_filter = n_filter
#         self.model = [nn.Conv2d(in_depth, self.n_filter, 3, 1, 1), nn.ReLU()]
#         for i in range(5):
#             self.model.append(
#                 nn.Conv2d(self.n_filter * (2 ** i), self.n_filter * (2 ** (i + 1)), 3, 2, 1))
#             self.model.append(nn.ReLU())
#         self.model = nn.Sequential(*self.model)
#         self.final_layers = nn.Sequential(*[nn.Linear(self.n_filter * (2 ** 5), 1), nn.Sigmoid()])
#
#     def forward(self, x):
#         outs = self.model(x)
#         outs = outs.squeeze()
#         return self.final_layers(outs)


class CycleDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, inputs):
        """Standard forward."""
        return self.model(inputs).squeeze()


class CycleGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3,
                 padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, inputs):
        """Standard forward"""
        return (self.model(inputs) + 1) / 2


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class CycleGAN:
    def __init__(self, X_depth, Y_depth, n_filter, lr, lmbda):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.g_X = CycleGenerator(Y_depth, X_depth).to(self.device)
        self.g_Y = CycleGenerator(X_depth, Y_depth).to(self.device)
        self.d_X = CycleDiscriminator(X_depth).to(self.device)
        self.d_Y = CycleDiscriminator(Y_depth).to(self.device)
        self.g_optimizer = torch.optim.Adam(list(self.g_X.parameters()) + list(self.g_Y.parameters()), lr=lr)
        self.d_optimizer = torch.optim.Adam(list(self.d_X.parameters()) + list(self.d_Y.parameters()), lr=lr / 2)
        self.lmbda = lmbda
        self.l1_loss = torch.nn.L1Loss()

    def train_d(self, x, y):
        x, y = x.to(self.device).float(), y.to(self.device).float()
        dx_loss = (self.d_X(x) - 1) ** 2 + self.d_X(self.g_X(y).detach()) ** 2
        dy_loss = (self.d_Y(y) - 1) ** 2 + self.d_Y(self.g_Y(x).detach()) ** 2

        loss = torch.mean(dx_loss + dy_loss)
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
        return loss.cpu().item()

    def train_g(self, x, y):
        x, y = x.to(self.device).float(), y.to(self.device).float()
        self.freeze(self.d_X)
        self.freeze(self.d_Y)

        dx_loss = torch.mean((self.d_X(self.g_X(y)) - 1) ** 2)
        dy_loss = torch.mean((self.d_Y(self.g_Y(x)) - 1) ** 2)

        cycle_loss1 = self.l1_loss(self.g_X(self.g_Y(x)), x)
        cycle_loss2 = self.l1_loss(self.g_Y(self.g_X(y)), y)
        cycle_loss = cycle_loss1 + cycle_loss2

        loss = dx_loss + dy_loss + self.lmbda * cycle_loss
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        self.reset(self.d_X)
        self.reset(self.d_Y)
        return loss.cpu().item()

    def translate(self, inputs, source_distribution):
        with torch.no_grad():
            if source_distribution == "X":
                self.g_Y.eval()
                x = inputs.to(self.device).float()
                y = self.g_Y(x)
                return y.permute(0, 2, 3, 1).cpu().numpy()
            else:
                self.g_X.eval()
                y = inputs.to(self.device).float()
                x = self.g_X(y)
                return x.permute(0, 2, 3, 1).cpu().numpy()

    def reconstruction(self, inputs, source_distribution):
        self.g_X.eval()
        self.g_Y.eval()
        with torch.no_grad():
            if source_distribution == "X":
                x = inputs.to(self.device).float()
                y = self.g_Y(x)
                x_ = self.g_X(y)
                return x_.permute(0, 2, 3, 1).cpu().numpy()
            else:
                y = inputs.to(self.device).float()
                x = self.g_X(y)
                y_ = self.g_Y(x)
                return y_.permute(0, 2, 3, 1).cpu().numpy()

    def freeze(self, sub_model):
        for param in sub_model.parameters():
            param.requires_grad = False

    def reset(self, sub_model):
        for param in sub_model.parameters():
            param.requires_grad = True


def q4(mnist_data, cmnist_data):
    """
    mnist_data: An (60000, 1, 28, 28) numpy array of black and white images with values in [0, 1]
    cmnist_data: An (60000, 3, 28, 28) numpy array of colored images with values in [0, 1]

    Returns
    - a (20, 28, 28, 1) numpy array of real MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of translated Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of reconstructed MNIST digits, in [0, 1]

    - a (20, 28, 28, 3) numpy array of real Colored MNIST digits, in [0, 1]
    - a (20, 28, 28, 1) numpy array of translated MNIST digits, in [0, 1]
    - a (20, 28, 28, 3) numpy array of reconstructed Colored MNIST digits, in [0, 1]
    """
    lr = 2e-4
    batch_size = 256
    k = 1
    lmbda = 10
    epochs = 20
    n_filters = 32
    model = CycleGAN(1, 3, n_filters, lr, lmbda)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    cmnist_loader = torch.utils.data.DataLoader(cmnist_data, batch_size=batch_size, shuffle=True)
    x_samples = torch.from_numpy(mnist_data[0:20])
    y_samples = torch.from_numpy(cmnist_data[0:20])
    for i in range(epochs):
        j = 0
        for x, y in zip(mnist_loader, cmnist_loader):
            g_loss = model.train_g(x, y)
            if j % k == 0:
                d_loss = model.train_d(x, y)
            j = j + 1
            print(f"{j}th epoch d_loss {d_loss} g_loss {g_loss}")

    x_translate = model.translate(x_samples, "X")
    x_reconstruct = model.reconstruction(x_samples, "X")
    y_translate = model.translate(y_samples, "Y")
    y_reconstruct = model.reconstruction(y_samples, "Y")
    x_samples = x_samples.permute(0, 2, 3, 1).numpy()
    y_samples = y_samples.permute(0, 2, 3, 1).numpy()
    return x_samples, x_translate, x_reconstruct, y_samples, y_translate, y_reconstruct


def q4_save():
    q4_save_results(q4)


if __name__ == '__main__':
    q1b()
