from deepul.hw2_helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm

np.random.seed(0)


def update_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] += 5e-4 / 500
        if param_group['lr'] >= 7e-4:
            param_group['lr'] = 7e-4


def train(model, train_loader, optimizer, epoch, grad_clip=None):
    model.train()

    train_losses = []
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    for x in train_loader:
        x = x + np.random.random(size=x.shape)
        x = 0.05 + (1 - 0.05) * (x / 4)
        x = x.to(device)
        loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append(loss.item())
        update_lr(optimizer, epoch)
    return train_losses


def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            loss = model.loss(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss / len(data_loader.dataset)

    return avg_loss.item()


def train_epochs(model, train_loader, test_loader, train_args, best_model=None, sample_list=[], inter_image=None):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = [eval_loss(model, test_loader)]
    print(f"initial test loss is {test_losses[0]}")
    best_loss = 1e8
    for epoch in range(epochs):
        model.train()
        train_losses.extend(train(model, train_loader, optimizer, epoch, grad_clip))
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            best_model.load_state_dict(model.state_dict())
        print(f'Epoch {epoch}, Test loss {test_loss:.4f}')
    print(f"best loss is {best_loss}")
    sample_list.append(best_model.sample())
    sample_list.append(best_model.interpolation())
    return train_losses, test_losses


def visualize_q1():
    visualize_q1_data(dset_type=1)
    visualize_q1_data(dset_type=2)


class MaskedLinear(torch.nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class Made(torch.nn.Module):
    def __init__(self, masked_matrix_list, hidden_units):
        super().__init__()
        self.fc1 = MaskedLinear(hidden_units[0], hidden_units[1])
        self.fc1.set_mask(masked_matrix_list[0])
        self.fc2 = MaskedLinear(hidden_units[1], hidden_units[2])
        self.fc2.set_mask(masked_matrix_list[1])
        self.fc3 = MaskedLinear(hidden_units[2], hidden_units[3])
        self.fc3.set_mask(masked_matrix_list[2])
        self.relu = torch.nn.ReLU()
        self.masked_matrix_list = masked_matrix_list

    def forward(self, inputs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        tmp = self.relu(self.fc1(inputs))
        tmp = self.relu(self.fc2(tmp))
        return self.fc3(tmp)


class ARFlow(nn.Module):
    def __init__(self, num_gauss, hidden_units):
        super().__init__()
        m0 = np.array([1, 2], dtype=np.int16)
        m1 = np.random.randint(low=1, high=3, size=hidden_units[1])
        m2 = np.random.randint(low=1, high=3, size=hidden_units[2])
        m3 = np.array([1, ] * (hidden_units[3] // 2) + [2, ] * (hidden_units[3] // 2), dtype=np.int16)
        M1 = torch.from_numpy(m1[np.newaxis].transpose() >= m0[np.newaxis])
        M2 = torch.from_numpy(m2[np.newaxis].transpose() >= m1[np.newaxis])
        M3 = torch.from_numpy(m3[np.newaxis].transpose() > m2[np.newaxis])
        self.num_gauss = num_gauss
        self.hidden_units = hidden_units
        self.nn = Made([M1, M2, M3], hidden_units)

    def forward(self, inputs):
        gauss_params = self.nn(inputs.float())
        first_dim_data = inputs[:, 0].unsqueeze(1).repeat(1, self.num_gauss)
        first_compo_pi = torch.softmax(gauss_params[:, 0:self.num_gauss], dim=1)
        first_compo_mu = gauss_params[:, self.num_gauss: 2 * self.num_gauss]
        first_compo_sigma = torch.exp(gauss_params[:, 2 * self.num_gauss: 3 * self.num_gauss])
        z0 = 0.5 * (1 + torch.erf((first_dim_data - first_compo_mu) * (1 / first_compo_sigma) / math.sqrt(2)))
        latent0 = torch.sum(first_compo_pi * z0, dim=1)
        z0 = 0
        z0_grad = (1 / (math.sqrt(2 * math.pi) * first_compo_sigma)) * torch.exp(
            -(((first_dim_data - first_compo_mu) / first_compo_sigma) ** 2) / 2)
        log_z0_grad = torch.log(torch.sum(first_compo_pi * z0_grad, dim=1))
        z0 = z0 + log_z0_grad

        second_dim_data = inputs[:, 1].unsqueeze(1).repeat(1, self.num_gauss)
        second_compo_pi = torch.softmax(gauss_params[:, 3 * self.num_gauss:4 * self.num_gauss], dim=1)
        second_compo_mu = gauss_params[:, 4 * self.num_gauss: 5 * self.num_gauss]
        second_compo_sigma = torch.exp(gauss_params[:, 5 * self.num_gauss: 6 * self.num_gauss])
        z1 = 0.5 * (1 + torch.erf((second_dim_data - second_compo_mu) * (1 / second_compo_sigma) / math.sqrt(2)))
        latent1 = torch.sum(second_compo_pi * z1, dim=1)
        z1 = 0
        z1_grad = (1 / (math.sqrt(2 * math.pi) * second_compo_sigma)) * torch.exp(
            -(((second_dim_data - second_compo_mu) / second_compo_sigma) ** 2) / 2)
        log_z1_grad = torch.log(torch.sum(second_compo_pi * z1_grad, dim=1))
        z1 = z1 + log_z1_grad
        return z0 + z1, torch.cat([latent0.unsqueeze(1), latent1.unsqueeze(1)], dim=1), log_z0_grad + log_z1_grad

    def loss(self, inputs):
        return -torch.mean(self(inputs)[0])

    def latents(self, inputs):
        return self(inputs)[1]

    def log_prob(self, inputs):
        return self(inputs)[2]

    def sample(self):
        pass


def q1_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets, or
               for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity).
        Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in [0,1]^2. This represents
        mapping the train set data points through our flow to the latent space.
    """

    """ YOUR CODE HERE """
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    num_gauss = 10
    model = ARFlow(num_gauss, [2, 100, 100, 6 * num_gauss]).to(device)
    best_model = ARFlow(num_gauss, [2, 100, 100, 6 * num_gauss]).to(device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader,
                                             dict(epochs=200, lr=1e-2), best_model)
    # heatmap
    dx, dy = 0.025, 0.025
    if dset_id == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    else:
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = torch.from_numpy(np.stack([x, y], axis=2).reshape(-1, 2)).float()
    densities = np.exp(model.log_prob(mesh_xs.to(device)).cpu().detach().numpy())

    latents = model.latents(torch.from_numpy(train_data).to(device).float()).cpu().detach().numpy()

    return train_losses, test_losses, densities, latents


def plot_q1a():
    '''
    bugs in
    1. the final layer of NN
    2. the expression of the pdf of the normal distribution
    3. the expression of the log probability
    '''
    q1_save_results(1, 'a', q1_a)
    q1_save_results(2, 'a', q1_a)


class RealNVP(nn.Module):
    def __init__(self, flow_num):
        super().__init__()
        self.affine_param_nn = nn.ModuleList([nn.Sequential(nn.Linear(1, 10),
                                                            nn.ReLU(),
                                                            nn.Linear(10, 10),
                                                            nn.ReLU(),
                                                            nn.Linear(10, 20),
                                                            nn.Tanh(),
                                                            nn.Linear(20, 2)) for _ in range(flow_num)])

    def forward(self, inputs):
        inputs = inputs.float()
        x0 = inputs[:, 0].unsqueeze(1)
        x1 = inputs[:, 1].unsqueeze(1)
        z0_old = x0
        z1_old = x1
        log_det_grad = []
        for affine in self.affine_param_nn:
            affine_scale = torch.exp(affine(z0_old)[:, 0].unsqueeze(1))
            affine_shift = affine(z0_old)[:, 1].unsqueeze(1)
            z0_new = z0_old
            z1_new = affine_scale * z1_old + affine_shift
            log_det_grad.append(torch.log(affine_scale))
            z0_old = z1_new
            z1_old = z0_new

        pz = ((1 / (math.sqrt(2 * math.pi))) * torch.exp(-(z0_new ** 2) / 2)) * (
                (1 / (math.sqrt(2 * math.pi))) * torch.exp(-(z1_new ** 2) / 2))
        log_pz = torch.log(pz)
        return log_pz + sum(log_det_grad), torch.cat((z0_new, z1_new), dim=1)

    def loss(self, inputs):
        return -torch.mean(self(inputs)[0])

    def latents(self, inputs):
        return self(inputs)[1]

    def log_prob(self, inputs):
        return self(inputs)[0]


def q1_b(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats in R^2
    test_data: An (n_test, 2) numpy array of floats in R^2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets, or
               for plotting a different region of densities

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (?,) of probabilities with values in [0, +infinity).
        Refer to the commented hint.
    - a numpy array of size (n_train, 2) of floats in R^2. This represents
        mapping the train set data points through our flow to the latent space.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    num_flow = 10
    model = RealNVP(num_flow).to(device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader,
                                             dict(epochs=200, lr=1e-2))
    # heatmap
    dx, dy = 0.025, 0.025
    if dset_id == 1:  # face
        x_lim = (-4, 4)
        y_lim = (-4, 4)
    else:
        x_lim = (-1.5, 2.5)
        y_lim = (-1, 1.5)
    y, x = np.mgrid[slice(y_lim[0], y_lim[1] + dy, dy),
                    slice(x_lim[0], x_lim[1] + dx, dx)]
    mesh_xs = torch.from_numpy(np.stack([x, y], axis=2).reshape(-1, 2)).float()
    densities = np.exp(model.log_prob(mesh_xs.to(device)).cpu().detach().numpy())
    latents = model.latents(torch.from_numpy(train_data).to(device).float()).cpu().detach().numpy()

    return train_losses, test_losses, densities, latents


def plot_q1b():
    '''
    bugs:
    the wrong function for pz, which should be the density function
    '''
    q1_save_results(1, 'b', q1_b)
    q1_save_results(2, 'b', q1_b)


def visualize_q2():
    visualize_q2_data()


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad):
        """2D Convolution with masked weight for Autoregressive connection"""
        super(MaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias=False)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()

        mask = torch.ones(ch_out, ch_in, height, width)
        mask[:, :, height // 2, width // 2:] = 0
        mask[:, :, height // 2 + 1:, :] = 0
        mask[:, :, height // 2, width // 2] = 1 if self.mask_type == 'B' else 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(torch.nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.maskA_block = torch.nn.Sequential(
            MaskedConv2d('A', channel_in, 64, 3, 1, 1),
            torch.nn.ReLU(),
            MaskedConv2d('A', 64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            MaskedConv2d('A', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
        )

        self.maskB_block = torch.nn.Sequential(
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            LayerNorm(64),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            LayerNorm(64),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            LayerNorm(64),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            LayerNorm(64),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            LayerNorm(64),
            torch.nn.ReLU()
        )

        self.one_one_conv = torch.nn.Sequential(
            MaskedConv2d('B', 64, 32, 1, 1, 0),
            torch.nn.ReLU(),
            MaskedConv2d('B', 32, channel_out, 1, 1, 0),
        )

        self.num_gauss = channel_out // 3

    def forward(self, inputs):
        inputs = inputs.float()
        outs = self.maskA_block(inputs.permute(0, 3, 1, 2))
        outs = self.maskB_block(outs)
        gauss_params = self.one_one_conv(outs).permute(0, 2, 3, 1)

        inputs = inputs.repeat(1, 1, 1, self.num_gauss)
        pi = torch.softmax(gauss_params[:, :, :, 0:self.num_gauss], dim=-1)
        mu = gauss_params[:, :, :, self.num_gauss: 2 * self.num_gauss]
        sigma = torch.exp(gauss_params[:, :, :, 2 * self.num_gauss: 3 * self.num_gauss])
        # this aims to compute the gradient of z: pdf
        z0_grad = (1 / (math.sqrt(2 * math.pi) * sigma)) * torch.exp(
            -(((inputs - mu) / sigma) ** 2) / 2)
        log_z0_grad = torch.log(torch.sum(pi * z0_grad, dim=-1))
        return log_z0_grad
        # return torch.sum(torch.sum(log_z0_grad, dim=-1), dim=-1)

    def cal_gauss_params(self, inputs):
        with torch.no_grad():
            device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            # inputs = inputs + torch.from_numpy(np.random.random(inputs.shape)).to(device)
            inputs = inputs.float()
            outs = self.maskA_block(inputs.permute(0, 3, 1, 2))
            outs = self.maskB_block(outs)
            gauss_params = self.one_one_conv(outs).permute(0, 2, 3, 1)
        return gauss_params

    def loss(self, inputs):
        return -torch.mean(self(inputs))

    def sample(self, image_shape=(20, 20)):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        images = torch.from_numpy(np.zeros((100, image_shape[0], image_shape[1]))).to(device)
        self.eval()
        with torch.no_grad():
            for i in range(image_shape[0]):
                for j in range(image_shape[1]):
                    gauss_params = self.cal_gauss_params(images.unsqueeze(-1))
                    pi = torch.softmax(gauss_params[:, i, j, 0: self.num_gauss], dim=-1)
                    component = torch.distributions.Categorical(pi).sample()
                    u = gauss_params[:, i, j, self.num_gauss: 2 * self.num_gauss]
                    u = u[np.arange(u.shape[0]), component]
                    sigma = torch.exp(gauss_params[:, i, j, 2 * self.num_gauss:3 * self.num_gauss])
                    sigma = sigma[np.arange(sigma.shape[0]), component]
                    images[:, i, j] = torch.clamp(
                        (u + sigma * torch.from_numpy(np.random.normal(0, 1, size=u.shape)).to(device)), 0, 2)
            print(images[0])
        return images.unsqueeze(-1).cpu().data.numpy()


def q2(train_data, test_data):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    H = W = 20
    Note that you should dequantize your train and test data, your dequantized pixels should all lie in [0,1]

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in [0, 1], where [0,0.5] represents a black pixel
        and [0.5,1] represents a white pixel. We will show your samples with and without noise.
    """

    """ YOUR CODE HERE """
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    num_gauss = 10
    model = PixelCNN(1, num_gauss * 3).to(device)
    best_model = PixelCNN(1, num_gauss * 3).to(device)
    sample = []
    train_data = train_data + np.random.normal(0, 0.5 / 3, size=train_data.shape)
    test_data = test_data + np.random.normal(0, 0.5 / 3, size=test_data.shape)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader, dict(epochs=0, lr=5e-4), best_model,
                                             sample)

    return train_losses, test_losses, sample[0]


def plot_q2():
    q2_save_results(q2)


class ResnetBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(n_filters, n_filters, 1, 1, 0),
                                 nn.ReLU(),
                                 nn.Conv2d(n_filters, n_filters, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Conv2d(n_filters, n_filters, 1, 1, 0))

    def forward(self, inputs):
        return self.net(inputs) + inputs


class SimpleResnet(nn.Module):
    def __init__(self, n_filters, n_block, n_in, n_out):
        super().__init__()
        self.init_conv = nn.Sequential(nn.Conv2d(n_in, n_filters, 3, 1, 1), nn.ReLU())
        self.res_block = nn.Sequential(*([ResnetBlock(n_filters)] * n_block))
        self.out_conv = nn.Sequential(nn.ReLU(), nn.Conv2d(n_filters, n_out, 3, 1, 1))

    def forward(self, inputs):
        outs = self.init_conv(inputs)
        outs = self.res_block(outs)
        return self.out_conv(outs)


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor


class ActNorm(nn.Module):
    def __init__(self, num_features, scale=1.):
        super().__init__()
        size = [1, 1, 1, num_features]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size), requires_grad=True))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size), requires_grad=True))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def initialize_parameters(self, inputs):
        if not self.training:
            return
        with torch.no_grad():
            bias = mean(inputs.clone(), dim=[0, 1, 2], keepdim=True)
            vars = mean((inputs.clone() - bias) ** 2, dim=[0, 1, 2], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def forward(self, inputs):
        if not self.inited:
            self.initialize_parameters(inputs)
        inputs = (inputs - self.bias) * torch.exp(self.logs)

        return inputs, self.logs

    def inverse_sample(self, z):
        self.eval()
        with torch.no_grad():
            z = z * torch.exp(-self.logs) + self.bias
        return z


class AffineCoupling(nn.Module):
    def __init__(self, n_filters, n_block, n_in, n_out, mask):
        super().__init__()
        self.nn = SimpleResnet(n_filters, n_block, n_in, n_out)
        self.register_buffer("mask", mask)

    def forward(self, inputs):
        inputs_ = inputs * self.mask
        log_s, t = torch.chunk(self.nn(inputs_.permute(0, 3, 1, 2).float()), 2, dim=1)
        t = t.permute(0, 2, 3, 1) * (1 - self.mask)
        log_s = log_s.permute(0, 2, 3, 1) * (1 - self.mask)
        z = inputs * torch.exp(log_s) + t
        log_det = log_s
        return z, log_det

    def inverse(self, z, inputs=None):
        self.eval()
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if inputs is None:
            inputs = torch.zeros(size=z.shape).to(device)
        with torch.no_grad():
            inputs_ = inputs * self.mask
            log_s, t = torch.chunk(self.nn(inputs_.permute(0, 3, 1, 2).float()), 2, dim=1)
            t = t.permute(0, 2, 3, 1) * (1 - self.mask)
            log_s = log_s.permute(0, 2, 3, 1) * (1 - self.mask)
            z = (z - t) * torch.exp(-log_s)
        return z

    def inverse_sample(self, z):
        self.eval()
        x = self.inverse(z)
        x = self.inverse(z, x)
        return x


class HighDimRealNVP(nn.Module):
    def __init__(self, image_shape, n_filters, n_block, n_in, n_out):
        super().__init__()
        self.image_shape = image_shape
        self.mask_checker = torch.from_numpy(
            np.array([i % 2 for i in range(image_shape[0] * image_shape[1])]).reshape(image_shape[0], image_shape[1]))
        self.mask_checker = self.mask_checker.unsqueeze(-1).repeat(1, 1, image_shape[-1])

        self.mask_channel = torch.cat(
            (torch.ones(image_shape[0] // 2, image_shape[1] // 2, 2),
             torch.zeros(image_shape[0] // 2, image_shape[1] // 2, 2)), dim=-1)
        self.mask_channel = self.mask_channel.repeat(1, 1, image_shape[-1])

        self.affine_coupling_checker1 = nn.ModuleList(
            [AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_checker),
             ActNorm(n_out // 2),
             AffineCoupling(n_filters, n_block, n_in, n_out, 1 - self.mask_checker),
             ActNorm(n_out // 2),
             AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_checker),
             ActNorm(n_out // 2),
             AffineCoupling(n_filters, n_block, n_in, n_out, 1 - self.mask_checker),
             ActNorm(n_out // 2)
             ])

        self.affine_coupling_channel = nn.ModuleList(
            [AffineCoupling(n_filters, n_block, 4 * n_in, 4 * n_out, self.mask_channel),
             ActNorm(2 * n_out),
             AffineCoupling(n_filters, n_block, 4 * n_in, 4 * n_out, 1 - self.mask_channel),
             ActNorm(2 * n_out),
             AffineCoupling(n_filters, n_block, 4 * n_in, 4 * n_out, self.mask_channel),
             ActNorm(2 * n_out)
             ])

        self.affine_coupling_checker2 = nn.ModuleList(
            [AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_checker),
             ActNorm(n_out // 2),
             AffineCoupling(n_filters, n_block, n_in, n_out, 1 - self.mask_checker),
             ActNorm(n_out // 2),
             AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_checker),
             ActNorm(n_out // 2)])

    def forward(self, inputs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        log_det = 0
        for layer in self.affine_coupling_checker1:
            inputs, tmp = layer(inputs)
            log_det = log_det + tmp
        inputs = self._squeeze_nxn(inputs)
        log_det = self._squeeze_nxn(log_det)

        for layer in self.affine_coupling_channel:
            inputs, tmp = layer(inputs)
            log_det = log_det + tmp
        inputs = self._unsqueeze_2x2(inputs)
        log_det = self._unsqueeze_2x2(log_det)

        for layer in self.affine_coupling_checker2:
            inputs, tmp = layer(inputs)
            log_det = log_det + tmp
        # outs = (1 / math.sqrt(2 * math.pi)) * torch.exp(
        #     -((inputs ** 2) / 2))
        outs = math.log(1 / math.sqrt(2 * math.pi)) - ((inputs ** 2) / 2)
        return outs + log_det

    def inverse(self, z):
        for layer in reversed(self.affine_coupling_checker2):
            z = layer.inverse_sample(z)
        z = self._squeeze_nxn(z)
        for layer in reversed(self.affine_coupling_channel):
            z = layer.inverse_sample(z)
        z = self._unsqueeze_2x2(z)
        for layer in reversed(self.affine_coupling_checker1):
            z = layer.inverse_sample(z)
        return z

    def sample(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        z = torch.from_numpy(np.random.normal(0, 1, size=(100, self.image_shape[0], self.image_shape[1], 3))).to(device)
        return torch.clamp((self.inverse(z) - 0.05) / 0.95, 0, 1).cpu().data.numpy()

    def loss(self, inputs):
        return (math.log(4 / 0.95) - torch.mean(self(inputs))) / math.log(2)

    def interpolation(self, inputs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        x = inputs
        for layer in self.affine_coupling_checker1:
            inputs, _ = layer(inputs)
        inputs = self._squeeze_nxn(inputs)

        for layer in self.affine_coupling_channel:
            inputs, _ = layer(inputs)
        inputs = self._unsqueeze_2x2(inputs)

        for layer in self.affine_coupling_checker2:
            inputs, _ = layer(inputs)

        image_list = []
        lamda = [0.8, 0.6, 0.4, 0.2]
        for i in range(5):
            image_list.append(x[2 * i].unsqueeze(0))
            for coeff in lamda:
                latent_code = coeff * inputs[2 * i].unsqueeze(0) + (1 - coeff) * inputs[2 * i + 1].unsqueeze(0)
                image_list.append(torch.clamp((self.inverse(latent_code) - 0.05) / 0.95, 0, 1))
            image_list.append(x[2 * i + 1].unsqueeze(0))
        return torch.cat(image_list, dim=0).cpu().data.numpy()

    def _squeeze_nxn(self, inputs, n_factor=2):
        """Squeezing operation: reshape to convert space to channels."""
        shape = inputs.shape
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        res = torch.reshape(
            inputs,
            [batch_size,
             height // n_factor,
             n_factor, width // n_factor,
             n_factor, channels])
        res = res.permute([0, 1, 3, 5, 2, 4])
        res = torch.reshape(
            res,
            [batch_size,
             height // n_factor,
             width // n_factor,
             channels * n_factor * n_factor])

        return res

    def _unsqueeze_2x2(self, inputs):
        """Unsqueezing operation: reshape to convert channels into space."""
        shape = inputs.shape
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        res = torch.reshape(inputs, [batch_size, height, width, channels // 4, 2, 2])
        res = res.permute([0, 1, 4, 2, 5, 3])
        res = torch.reshape(res, [batch_size, 2 * height, 2 * width, channels // 4])
        return res


def visualize_q3():
    visualize_q3_data()


def q3_a(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    model = HighDimRealNVP((train_data.shape[1], train_data.shape[2], 3), 128, 8, 3, 6)
    best_model = HighDimRealNVP((train_data.shape[1], train_data.shape[2], 3), 128, 8, 3, 6)

    model = model.to(device)
    best_model = best_model.to(device)

    # model = nn.DataParallel(model).module
    # best_model = nn.DataParallel(best_model).module

    sample = []

    alpha = 0.05
    inter_image = train_data[10:20]
    # train_data = train_data + np.random.random(size=train_data.shape)
    # train_data = alpha + (1 - alpha) * (train_data / 4)
    test_data = test_data + np.random.random(size=test_data.shape)
    test_data = alpha + (1 - alpha) * (test_data / 4)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=48, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=48)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader,
                                             dict(epochs=120, lr=0, grad_clip=10, decay=0.95, end=1e-8), best_model,
                                             sample)

    return train_losses, test_losses, sample[0], model.interpolation(torch.from_numpy(alpha + (1 - alpha) * (inter_image / 4)).float())


def plot_q3a():
    q3_save_results(q3_a, 'a')


class BadHighDimRealNVP(nn.Module):
    def __init__(self, image_shape, n_filters, n_block, n_in, n_out):
        super().__init__()
        self.image_shape = image_shape
        self.mask_top = torch.from_numpy(np.zeros(shape=image_shape))
        self.mask_top[0:(self.image_shape[0] // 2)] = 1

        self.mask_bottom = torch.from_numpy(np.zeros(shape=image_shape))
        self.mask_bottom[(self.image_shape[0] // 2):] = 1

        self.mask_left = torch.from_numpy(np.zeros(shape=image_shape))
        self.mask_left[:, 0:(self.image_shape[1] // 2)] = 1

        self.mask_right = torch.from_numpy(np.zeros(shape=image_shape))
        self.mask_right[:, (self.image_shape[1] // 2):] = 1

        self.affine_coupling_checker1 = nn.ModuleList(
            [AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_top),
             AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_bottom),
             AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_left),
             AffineCoupling(n_filters, n_block, n_in, n_out, self.mask_right)] * 2)

    def forward(self, inputs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        log_det = 0
        for layer in self.affine_coupling_checker1:
            inputs, tmp = layer(inputs)
            log_det = log_det + tmp
        outs = math.log(1 / math.sqrt(2 * math.pi)) - ((inputs ** 2) / 2)
        return outs + log_det

    def inverse(self, z):
        for layer in reversed(self.affine_coupling_checker1):
            z = layer.inverse_sample(z)
        return z

    def sample(self):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        z = torch.from_numpy(np.random.normal(0, 1, size=(100, self.image_shape[0], self.image_shape[1], 3))).to(device)
        return torch.clamp((self.inverse(z) - 0.05) / 0.95, 0, 1).cpu().data.numpy()

    def loss(self, inputs):
        return (math.log(4 / 0.95) - torch.mean(self(inputs))) / math.log(2)

    def interpolation(self, inputs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        x = inputs
        for layer in self.affine_coupling_checker1:
            inputs, _ = layer(inputs)

        image_list = []
        lamda = [0.8, 0.6, 0.4, 0.2]
        for i in range(5):
            image_list.append(x[2 * i].unsqueeze(0).float())
            for coeff in lamda:
                latent_code = coeff * inputs[2 * i].unsqueeze(0) + (1 - coeff) * inputs[2 * i + 1].unsqueeze(0)
                image_list.append(torch.clamp((self.inverse(latent_code) - 0.05) / 0.95, 0, 1).float())
            image_list.append(x[2 * i + 1].unsqueeze(0).float())
        return torch.cat(image_list, dim=0).cpu().data.numpy()


def q3_b(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

    model = BadHighDimRealNVP((train_data.shape[1], train_data.shape[2], 3), 128, 8, 3, 6)
    best_model = BadHighDimRealNVP((train_data.shape[1], train_data.shape[2], 3), 128, 8, 3, 6)

    model = model.to(device)
    best_model = best_model.to(device)

    sample = []

    alpha = 0.05
    inter_image = train_data[10:20]
    # train_data = train_data + np.random.random(size=train_data.shape)
    # train_data = alpha + (1 - alpha) * (train_data / 4)
    test_data = test_data + np.random.random(size=test_data.shape)
    test_data = alpha + (1 - alpha) * (test_data / 4)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    train_losses, test_losses = train_epochs(model, train_loader, test_loader, dict(epochs=120, lr=0, grad_clip=1),
                                             best_model,
                                             sample, torch.from_numpy(alpha + (1 - alpha) * (inter_image / 4)).float())

    return train_losses, test_losses, sample[0], sample[1]


def plot_q3b():
    q3_save_results(q3_b, 'b')


def q4_a(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

    """ YOUR CODE HERE """


def plot_q4a():
    q3_save_results(q4_a, 'bonus_a')


def q4_b(train_data, test_data):
    """
    train_data: A (n_train, H, W, 3) uint8 numpy array of quantized images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, 3) uint8 numpy array of binary images with values in {0, 1, 2, 3}

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 3) of samples with values in [0, 1]
    - a numpy array of size (30, H, W, 3) of interpolations with values in [0, 1].
    """

    """ YOUR CODE HERE """


def plot_q4b():
    q3_save_results(q4_b, 'bonus_b')


if __name__ == '__main__':
    plot_q3b()
