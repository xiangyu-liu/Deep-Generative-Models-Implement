from deepul.hw3_helper import *
import torch
from torch.distributions.normal import Normal
from torch import nn
import numpy as np

np.random.seed(0)


def train(model, train_loader, optimizer, grad_clip=None):
    model.train()
    train_losses = []
    for x in train_loader:
        loss = model.loss(x)
        elbo = loss[0]
        optimizer.zero_grad()
        elbo.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append([loss[0].cpu().item(), loss[1].cpu().item(), loss[2].cpu().item()])
    return train_losses


def eval_loss(model, data_loader):
    model.eval()
    total_elbo = 0
    total_reconstruction = 0
    total_kl = 0
    for x in data_loader:
        with torch.no_grad():
            loss = model.loss(x)
            total_elbo += loss[0].cpu().item() * x.shape[0]
            total_reconstruction += loss[1].cpu().item() * x.shape[0]
            total_kl += loss[2].cpu().item() * x.shape[0]
    avg_elbo = total_elbo / len(data_loader.dataset)
    avg_reconstruction = total_reconstruction / len(data_loader.dataset)
    avg_kl = total_kl / len(data_loader.dataset)

    return [avg_elbo, avg_reconstruction, avg_kl]


def train_epochs(model, train_loader, test_loader, train_args, best_model=None):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 1e8
    test_losses = [eval_loss(model, test_loader)]
    train_losses = []
    # test_losses = [[0, 0, 0]]
    # train_losses = [[0, 0, 0]]
    for epoch in range(epochs):
        train_losses.extend(train(model, train_loader, optimizer, grad_clip))
        test_losses.append(eval_loss(model, test_loader))
        if test_losses[-1][0] < best_loss:
            best_loss = test_losses[-1][0]
            best_model.load_state_dict(model.state_dict())
        print(f'Epoch {epoch}, ELBO {test_losses[-1][0]:.4f}')
    return train_losses, test_losses


def visual_q1a():
    visualize_q1_data('a', 1)
    visualize_q1_data('a', 2)


class VAE(nn.Module):
    def __init__(self, sample_num, encoder_units, decoder_units):
        super().__init__()
        self.sample_num = sample_num
        self.data_dim = encoder_units[0]
        self.latent_dim = decoder_units[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder = []
        for i in range(len(encoder_units) - 1):
            encoder.append(nn.Linear(encoder_units[i], encoder_units[i + 1]))
            encoder.append(nn.ReLU())
        encoder.pop()
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        for i in range(len(decoder_units) - 1):
            decoder.append(nn.Linear(decoder_units[i], decoder_units[i + 1]))
            decoder.append(nn.ReLU())
        decoder.pop()
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.to(self.device)
        # Todo: check the dimension
        x = x.unsqueeze(1).repeat(1, self.sample_num, 1)
        u_z, log_sigma_z = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        sigma_z = torch.exp(log_sigma_z)
        epsilon_samples = torch.normal(0, 1, size=(batch_size, self.sample_num, self.latent_dim)).to(self.device)
        z_samples = u_z + sigma_z * epsilon_samples

        u_x, log_sigma_x = torch.chunk(self.decoder(z_samples), chunks=2, dim=-1)
        sigma_x = torch.exp(log_sigma_x)

        log_p_theta = Normal(u_x, sigma_x).log_prob(x)
        log_p_z = Normal(0, 1).log_prob(z_samples)
        log_q_phi = Normal(u_z, sigma_z).log_prob(z_samples)
        return log_p_theta.mean(), log_p_z.mean(), log_q_phi.mean()

    def loss(self, x):
        losses = self(x)
        return (losses[2] - losses[1] - losses[0]), -losses[0], losses[2] - losses[1]

    def sample(self):
        with torch.no_grad():
            z = torch.normal(0, 1, size=(1000, self.latent_dim)).to(self.device)
            u_x, log_sigma_x = torch.chunk(self.decoder(z), chunks=2, dim=1)
            sigma_x = torch.exp(log_sigma_x)
            epsilon_sample = torch.normal(0, 1, size=(1000, self.data_dim)).to(self.device)
            return epsilon_sample * sigma_x + u_x, u_x


def q1(train_data, test_data, part, dset_id):
    """
    train_data: An (n_train, 2) numpy array of floats
    test_data: An (n_test, 2) numpy array of floats

    (You probably won't need to use the two inputs below, but they are there
     if you want to use them)
    part: An identifying string ('a' or 'b') of which part is being run. Most likely
          used to set different hyperparameters for different datasets
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a numpy array of size (1000, 2) of 1000 samples WITH decoder noise, i.e. sample z ~ p(z), x ~ p(x|z)
    - a numpy array of size (1000, 2) of 1000 samples WITHOUT decoder noise, i.e. sample z ~ p(z), x = mu(z)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    sample_num = 10
    data_dim = 2
    latent_dim = 2
    encoder_units = [data_dim, 20, 20, 2 * latent_dim]
    decoder_units = [latent_dim, 20, 20, 2 * data_dim]
    model = VAE(sample_num, encoder_units, decoder_units).to(device)
    best_model = VAE(sample_num, encoder_units, decoder_units).to(device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    train_losses, test_losses = train_epochs(model, train_loader, test_loader, dict(epochs=100, lr=5e-4), best_model)
    sample_noise, sample_no_noise = best_model.sample()
    return 2 * np.array(train_losses), 2 * np.array(test_losses), sample_noise, sample_no_noise


def q1a():
    q1_save_results('a', 1, q1)
    q1_save_results('a', 2, q1)


def visual_q1b():
    visualize_q1_data('b', 1)
    visualize_q1_data('b', 2)


def q1b():
    q1_save_results('b', 1, q1)
    q1_save_results('b', 2, q1)


def visual_q2a():
    visualize_svhn()
    visualize_cifar10()


class ImageVAE(nn.Module):
    def __init__(self, sample_num, latend_dim):
        super().__init__()
        self.sample_num = sample_num
        self.latent_dim = latend_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_conv = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 64, 3, 2, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 128, 3, 2, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 256, 3, 2, 1),
                                          nn.ReLU())
        self.encoder_linear = nn.Linear(4 * 4 * 256, 2 * self.latent_dim)
        self.decoder_linear = nn.Sequential(nn.Linear(self.latent_dim, 4 * 4 * 128), nn.ReLU())

        self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 3, 3, 1, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2).to(self.device).float()
        outs = self.encoder_conv(x)
        outs = outs.view(size=(batch_size, -1))
        u_z, log_sigma_z = torch.chunk(self.encoder_linear(outs), 2, dim=-1)
        u_z = u_z.unsqueeze(1).repeat(1, self.sample_num, 1)
        sigma_z = torch.exp(log_sigma_z).unsqueeze(1).repeat(1, self.sample_num, 1)
        epsilon = torch.normal(0, 1, size=(batch_size, self.sample_num, self.latent_dim)).to(self.device)
        z_samples = u_z + epsilon * sigma_z

        z_samples = z_samples.view(size=(-1, self.latent_dim))
        z_outs = self.decoder_linear(z_samples)
        z_outs = z_outs.view(size=(batch_size * self.sample_num, 128, 4, 4))
        z_outs = self.decoder_conv(z_outs)
        z_outs = z_outs.view(batch_size, self.sample_num, 3, 32, 32)
        log_p_theta = Normal(z_outs, 1).log_prob(x.unsqueeze(1).repeat(1, self.sample_num, 1, 1, 1))
        log_p_z = Normal(0, 1).log_prob(z_samples).view(batch_size, self.sample_num, -1)
        log_q_phi = Normal(u_z, sigma_z).log_prob(z_samples.view(size=u_z.shape))
        mse_loss = (z_outs - x.unsqueeze(1).repeat(1, self.sample_num, 1, 1, 1)) ** 2

        return log_p_theta.sum(dim=[2, 3, 4]), log_p_z.sum(dim=[2]), log_q_phi.sum(dim=[2]), mse_loss.sum(
            dim=[2, 3, 4]), z_outs

    def loss(self, x):
        losses = self(x)
        elbo = torch.mean(losses[2] - losses[1] + losses[3])
        reconstruction_loss = torch.mean(losses[3])
        kl_loss = torch.mean(losses[2] - losses[1])
        return elbo, reconstruction_loss, kl_loss

    def sample(self):
        with torch.no_grad():
            z_samples = torch.normal(0, 1, size=(100, self.latent_dim)).to(self.device)
            z_outs = self.decoder_linear(z_samples)
            z_outs = z_outs.view(size=(100, 128, 4, 4))
            z_outs = self.decoder_conv(z_outs)
            return z_outs.permute(0, 2, 3, 1).cpu().data.numpy()

    def sample_pairs(self, x):
        pre_sample_num = self.sample_num
        self.sample_num = 1
        x = torch.from_numpy(x).to(self.device).float()
        with torch.no_grad():
            results = self(x)
            z_outs = torch.squeeze(results[-1])
            reconstructed_images = z_outs
        self.sample_num = pre_sample_num
        image_list = []
        for i in range(x.shape[0]):
            image_list.extend([x[i], reconstructed_images[i].permute(1, 2, 0)])
        return torch.stack(image_list, dim=0).cpu().data.numpy()

    def interpolation(self, x):
        batch_size = x.shape[0]
        x = torch.from_numpy(x)
        x = x.permute(0, 3, 1, 2).to(self.device).float()
        outs = self.encoder_conv(x)
        outs = outs.view(size=(batch_size, -1))
        u_z, _ = torch.chunk(self.encoder_linear(outs), 2, dim=-1)

        image_list = []
        lamda = [(i + 1) / 9 for i in range(8)]
        for i in range(10):
            image_list.append(x[2 * i].permute(1, 2, 0))
            for coeff in lamda:
                z_samples = (1 - coeff) * u_z[2 * i] + coeff * u_z[2 * i + 1]
                z_outs = self.decoder_linear(z_samples)
                z_outs = z_outs.view(size=(1, 128, 4, 4))
                z_outs = self.decoder_conv(z_outs)
                z_outs = z_outs.view(1, 3, 32, 32)
                image_list.append(z_outs[0].permute(1, 2, 0))
            image_list.append(x[2 * i + 1].permute(1, 2, 0))
        return torch.stack(image_list, dim=0).cpu().data.numpy()


def q2_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    sample_num = 2
    latent_dim = 16
    model = ImageVAE(sample_num, latent_dim).to(device)
    best_model = ImageVAE(sample_num, latent_dim).to(device)
    train_data = (train_data / 255 - 0.5) * 2
    test_data = (test_data / 255 - 0.5) * 2
    images_set1 = train_data[0:50]
    images_set2 = train_data[0:20]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    train_losses, test_losses = train_epochs(model, train_loader, test_loader, dict(epochs=50, lr=1e-3), best_model)
    best_model.eval()
    image_samples = (best_model.sample() + 1) * 255 / 2
    image_pairs = (best_model.sample_pairs(images_set1) + 1) * 255 / 2
    image_interpolation = (best_model.interpolation(images_set2) + 1) * 255 / 2
    return np.array(train_losses), np.array(test_losses), image_samples, image_pairs, image_interpolation


def q2a():
    q2_save_results('a', 1, q2_a)
    q2_save_results('a', 2, q2_a)


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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.register_parameter("shift", nn.Parameter(torch.ones(size=(1, hidden_units[-1] // 2)), requires_grad=True))
        self.register_parameter("scale", nn.Parameter(torch.ones(size=(1, hidden_units[-1] // 2)), requires_grad=True))
        self.fc1 = MaskedLinear(hidden_units[0], hidden_units[1])
        self.fc1.set_mask(masked_matrix_list[0])
        self.fc2 = MaskedLinear(hidden_units[1], hidden_units[2])
        self.fc2.set_mask(masked_matrix_list[1])
        self.fc3 = MaskedLinear(hidden_units[2], hidden_units[3])
        self.fc3.set_mask(masked_matrix_list[2])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.masked_matrix_list = masked_matrix_list
        self.latent_dim = hidden_units[-1] // 2

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        outs = self.relu(self.fc1(inputs))
        outs = self.relu(self.fc2(outs))
        log_s, t = torch.chunk(self.relu(self.fc3(outs)), chunks=2, dim=-1)
        log_s = self.scale * self.tanh(log_s) + self.shift
        epsilon = inputs * torch.exp(log_s) + t
        log_p_epsilon = Normal(0, 1).log_prob(epsilon)
        log_det = log_s
        return log_p_epsilon + log_det

    def sample(self, sample_num):
        z = torch.normal(0, 1, size=(sample_num, self.latent_dim))
        x = torch.zeros(size=z.shape)
        for _ in range(self.latent_dim):
            x = self.inverse(z, x)
        return x

    def inverse(self, z, x):
        z = z.to(self.device)
        x = x.to(self.device)
        outs = self.relu(self.fc1(x))
        outs = self.relu(self.fc2(outs))
        log_s, t = torch.chunk(self.relu(self.fc3(outs)), chunks=2, dim=-1)
        log_s = self.scale * self.tanh(log_s) + self.shift
        x = (z - t) / torch.exp(log_s)
        return x


class ImageVAEAFPrior(nn.Module):
    def __init__(self, sample_num, latent_dim):
        super().__init__()
        self.sample_num = sample_num
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_conv = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 64, 3, 2, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(64, 128, 3, 2, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(128, 256, 3, 2, 1),
                                          nn.ReLU())
        self.encoder_linear = nn.Linear(4 * 4 * 256, 2 * self.latent_dim)
        self.decoder_linear = nn.Sequential(nn.Linear(self.latent_dim, 4 * 4 * 128), nn.ReLU())
        self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                          nn.ReLU(),
                                          nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                          nn.ReLU(),
                                          nn.Conv2d(32, 3, 3, 1, 1), nn.Tanh())
        hidden_units = [latent_dim, 256, 256, 2 * latent_dim]
        m0 = np.array([i + 1 for i in range(latent_dim)], dtype=np.int8)
        m1 = np.random.randint(low=1, high=latent_dim, size=hidden_units[1])
        m2 = np.random.randint(low=1, high=latent_dim, size=hidden_units[2])
        m3 = np.array([i + 1 for i in range(latent_dim)] * 2, dtype=np.int8)
        M1 = torch.from_numpy(m1[np.newaxis].transpose() >= m0[np.newaxis])
        M2 = torch.from_numpy(m2[np.newaxis].transpose() >= m1[np.newaxis])
        M3 = torch.from_numpy(m3[np.newaxis].transpose() > m2[np.newaxis])
        self.af_flow = Made([M1, M2, M3], hidden_units)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2).to(self.device).float()
        outs = self.encoder_conv(x)
        outs = outs.view(size=(batch_size, -1))
        u_z, log_sigma_z = torch.chunk(self.encoder_linear(outs), 2, dim=-1)
        u_z = u_z.unsqueeze(1).repeat(1, self.sample_num, 1)
        sigma_z = torch.exp(log_sigma_z).unsqueeze(1).repeat(1, self.sample_num, 1)
        epsilon = torch.normal(0, 1, size=(batch_size, self.sample_num, self.latent_dim)).to(self.device)
        z_samples = u_z + epsilon * sigma_z

        z_samples = z_samples.view(size=(-1, self.latent_dim))
        z_outs = self.decoder_linear(z_samples)
        z_outs = z_outs.view(size=(batch_size * self.sample_num, 128, 4, 4))
        z_outs = self.decoder_conv(z_outs)
        z_outs = z_outs.view(batch_size, self.sample_num, 3, 32, 32)
        log_p_theta = Normal(z_outs, 1).log_prob(x.unsqueeze(1).repeat(1, self.sample_num, 1, 1, 1))
        log_p_z = self.af_flow.forward(z_samples).view(batch_size, self.sample_num, -1)
        log_q_phi = Normal(u_z, sigma_z).log_prob(z_samples.view(size=u_z.shape))
        mse_loss = (z_outs - x.unsqueeze(1).repeat(1, self.sample_num, 1, 1, 1)) ** 2

        return log_p_theta.sum(dim=[2, 3, 4]), log_p_z.sum(dim=[2]), log_q_phi.sum(dim=[2]), mse_loss.sum(
            dim=[2, 3, 4]), z_outs

    def loss(self, x):
        losses = self(x)
        elbo = torch.mean(losses[2] - losses[1] + losses[3])
        reconstruction_loss = torch.mean(losses[3])
        kl_loss = torch.mean(losses[2] - losses[1])
        return elbo, reconstruction_loss, kl_loss

    def sample(self):
        with torch.no_grad():
            z_samples = self.af_flow.sample(100).to(self.device)
            z_outs = self.decoder_linear(z_samples)
            z_outs = z_outs.view(size=(100, 128, 4, 4))
            z_outs = self.decoder_conv(z_outs)
            return z_outs.permute(0, 2, 3, 1).cpu().data.numpy()

    def sample_pairs(self, x):
        pre_sample_num = self.sample_num
        self.sample_num = 1
        x = torch.from_numpy(x).to(self.device).float()
        with torch.no_grad():
            results = self(x)
            z_outs = torch.squeeze(results[-1])
            reconstructed_images = z_outs
        self.sample_num = pre_sample_num
        image_list = []
        for i in range(x.shape[0]):
            image_list.extend([x[i], reconstructed_images[i].permute(1, 2, 0)])
        return torch.stack(image_list, dim=0).cpu().data.numpy()

    def interpolation(self, x):
        batch_size = x.shape[0]
        x = torch.from_numpy(x)
        x = x.permute(0, 3, 1, 2).to(self.device).float()
        outs = self.encoder_conv(x)
        outs = outs.view(size=(batch_size, -1))
        u_z, _ = torch.chunk(self.encoder_linear(outs), 2, dim=-1)

        image_list = []
        lamda = [(i + 1) / 9 for i in range(8)]
        for i in range(10):
            image_list.append(x[2 * i].permute(1, 2, 0))
            for coeff in lamda:
                z_samples = (1 - coeff) * u_z[2 * i] + coeff * u_z[2 * i + 1]
                z_outs = self.decoder_linear(z_samples)
                z_outs = z_outs.view(size=(1, 128, 4, 4))
                z_outs = self.decoder_conv(z_outs)
                z_outs = z_outs.view(1, 3, 32, 32)
                image_list.append(z_outs[0].permute(1, 2, 0))
            image_list.append(x[2 * i + 1].permute(1, 2, 0))
        return torch.stack(image_list, dim=0).cpu().data.numpy()


def q2_b(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in {0, ..., 255}
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples from your VAE with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in {0, ..., 255}
    - a (100, 32, 32, 3) numpy array of 10 interpolations of length 10 between
      pairs of test images. The output should be those 100 images flattened into
      the specified shape with values in {0, ..., 255}
    """

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    sample_num = 2
    latent_dim = 16
    model = ImageVAEAFPrior(sample_num, latent_dim).to(device)
    best_model = ImageVAEAFPrior(sample_num, latent_dim).to(device)
    train_data = (train_data / 255 - 0.5) * 2
    test_data = (test_data / 255 - 0.5) * 2
    images_set1 = train_data[0:50]
    images_set2 = train_data[0:20]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    train_losses, test_losses = train_epochs(model, train_loader, test_loader, dict(epochs=50, lr=1e-3), best_model)
    best_model.eval()
    image_samples = (best_model.sample() + 1) * 255 / 2
    image_pairs = (best_model.sample_pairs(images_set1) + 1) * 255 / 2
    image_interpolation = (best_model.interpolation(images_set2) + 1) * 255 / 2
    return np.array(train_losses), np.array(test_losses), image_samples, image_pairs, image_interpolation


def q2b():
    q2_save_results('b', 1, q2_b)
    q2_save_results('b', 2, q2_b)


def vq_train(model, train_loader, optimizer, grad_clip=None):
    model.train()
    train_losses = []
    for x in train_loader:
        loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        train_losses.append(loss.cpu().item())
    return train_losses


def vq_eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    for x in data_loader:
        with torch.no_grad():
            loss = model.loss(x)
            total_loss += loss.cpu().item() * x.shape[0]
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def vq_train_epochs(model, train_loader, test_loader, train_args, best_model=None):
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 1e8
    test_losses = [vq_eval_loss(model, test_loader)]
    print(f"initial test loss is {test_losses[0]}")
    train_losses = []
    # test_losses = [0,]
    # train_losses = [0,]
    for epoch in range(epochs):
        train_losses.extend(vq_train(model, train_loader, optimizer, grad_clip))
        test_losses.append(vq_eval_loss(model, test_loader))
        if test_losses[-1] < best_loss:
            best_loss = test_losses[-1]
            best_model.load_state_dict(model.state_dict())
        print(f'Epoch {epoch}, loss {test_losses[-1]:.4f}')
    print(f"best loss is {best_loss}")
    return train_losses, test_losses


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


class MaskedConv2dBlock(nn.Module):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad):
        super().__init__()
        self.mask_conv = nn.Sequential(*[MaskedConv2d(mask_type, c_in, c_out, k_size, stride, pad)] * 2)

    def forward(self, x):
        return self.mask_conv(x) + x


class ResidualBlocks(nn.Module):
    def __init__(self, kernel_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.net = nn.Sequential(
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
            nn.Conv2d(kernel_num, kernel_num, 3, 1, 1),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
            nn.Conv2d(kernel_num, kernel_num, 3, 1, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        return self.net(x) + x


class PixelCNN(torch.nn.Module):
    def __init__(self, channel_in, channel_out, vq_vae):
        super().__init__()
        self.vae = vq_vae
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = nn.Embedding(channel_out, 64)
        self.maskA_block = torch.nn.Sequential(
            MaskedConv2d('A', 64, 64, 7, 1, 3),
            LayerNorm(64),
            torch.nn.ReLU()
        )

        self.maskB_block = torch.nn.Sequential(*[MaskedConv2d('B', 64, 64, 7, 1, 3),
                                                 LayerNorm(64),
                                                 torch.nn.ReLU()] * 4)

        self.one_one_conv = torch.nn.Sequential(
            MaskedConv2d('B', 64, 128, 1, 1, 0),
            LayerNorm(128),
            torch.nn.ReLU(),
            MaskedConv2d('B', 128, channel_out, 1, 1, 0),
        )
        self.criteria = nn.CrossEntropyLoss()
        self.latent_num = channel_out

    def forward(self, x):
        x = x.to(self.device).float()
        self.vae.eval()
        with torch.no_grad():
            zq_index = self.vae(x)[-2]
        outs = self.embedding(zq_index).permute(0, 3, 1, 2)
        outs = self.maskA_block(outs)
        outs = self.maskB_block(outs)
        outs = self.one_one_conv(outs)
        return self.criteria(outs, zq_index)

    def loss(self, x):
        return self(x)

    def sample(self):
        self.vae.eval()
        self.eval()
        with torch.no_grad():
            zq_index = torch.zeros((100, 8, 8)).to(self.device)
            for j in range(8):
                for k in range(8):
                    outs = self.embedding(zq_index.long()).permute(0, 3, 1, 2)
                    outs = self.maskA_block(outs)
                    outs = self.maskB_block(outs)
                    outs = self.one_one_conv(outs)
                    outs = torch.softmax(outs, dim=1)
                    cate_distr = torch.distributions.Categorical(outs.permute(0, 2, 3, 1).reshape(-1, self.latent_num))
                    zq_index.data[:, j, k] = cate_distr.sample().reshape(outs.shape[0], 8, 8)[:, j, k]

            return self.vae.sample(zq_index).permute(0, 2, 3, 1).cpu().numpy()

    def to_one_hot(self, zq_index_):
        zq_index = torch.flatten(zq_index_)
        zq_index = torch.zeros(size=(zq_index.shape[0], self.latent_num)).to(self.device)
        zq_index.zero_()
        zq_index.scatter_(1, torch.flatten(zq_index_).unsqueeze(1), 1)
        zq_index = zq_index.reshape(shape=list(zq_index_.shape) + [-1, ]).float()
        return zq_index.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    def __init__(self, latents_dim, latents_num):
        super().__init__()
        self.latents_dim = latents_dim
        self.latents_num = latents_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(nn.Conv2d(3, latents_dim, 4, 2, 1),
                                     nn.BatchNorm2d(latents_dim),
                                     nn.ReLU(),
                                     nn.Conv2d(latents_dim, latents_dim, 4, 2, 1),
                                     ResidualBlocks(latents_dim),
                                     ResidualBlocks(latents_dim)
                                     )
        self.decoder = nn.Sequential(ResidualBlocks(latents_dim),
                                     ResidualBlocks(latents_dim),
                                     nn.BatchNorm2d(latents_dim),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(latents_dim, latents_dim, 4, 2, 1),
                                     nn.BatchNorm2d(latents_dim),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(latents_dim, 3, 4, 2, 1))
        code_book = 2 * (torch.rand(size=(latents_num, latents_dim), requires_grad=True) - 0.5) / latents_num
        self.register_parameter("code_book", nn.Parameter(code_book, requires_grad=True))

    def forward(self, x):
        x = x.to(self.device).permute(0, 3, 1, 2).float()
        batch_size = x.shape[0]
        ze = self.encoder(x)
        ze_repeated = ze.unsqueeze(-1).repeat(1, 1, 1, 1, self.latents_num).permute(0, 2, 3, 4, 1)
        code_book_repeated = self.code_book.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size, 8, 8, 1, 1)
        distance = torch.sum((ze_repeated - code_book_repeated) ** 2, dim=-1)
        q_index = torch.argmin(distance, dim=-1)
        zq_with_grad = self.code_book[q_index].permute(0, 3, 1, 2)
        zq = ze + (zq_with_grad - ze).detach()
        outs = self.decoder(zq)
        recon_loss = (x - outs) ** 2
        vq_loss = (ze.detach() - zq_with_grad) ** 2
        commit_loss = (ze - zq_with_grad.detach()) ** 2
        return torch.mean(recon_loss), torch.mean(vq_loss), torch.mean(commit_loss), q_index.long(), outs

    def loss(self, x):
        losses = self(x)
        return torch.mean(losses[0] + losses[1] + losses[2])

    def sample(self, zq_index):
        zq = self.code_book[zq_index.long()].permute(0, 3, 1, 2)
        with torch.no_grad():
            return self.decoder(zq)

    def sample_pairs(self, x):
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            results = self(x)
            reconstructed_images = results[-1]
        image_list = []
        for i in range(x.shape[0]):
            image_list.extend([x[i].float(), reconstructed_images[i].permute(1, 2, 0)])
        return torch.stack(image_list, dim=0).cpu().numpy()


def q3(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples (an equal number from each class) with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in [0, 255]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    latents_dim = 256
    latents_num = 128
    model = VQVAE(latents_dim, latents_num).to(device)
    best_model = VQVAE(latents_dim, latents_num).to(device)
    pixel_cnn = PixelCNN(64, latents_num, best_model).to(device)
    best_pixel_cnn = PixelCNN(64, latents_num, best_model).to(device)

    train_data = (train_data / 255 - 0.5) * 2
    test_data = (test_data / 255 - 0.5) * 2
    images_set1 = train_data[0:50]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    train_losses, test_losses = vq_train_epochs(model, train_loader, test_loader, dict(epochs=35, lr=1e-3), best_model)
    best_model.eval()
    image_pairs = np.clip((best_model.sample_pairs(images_set1) + 1) * 255 / 2, 0, 255)

    train_pixel_cnn, test_pixel_cnn = vq_train_epochs(pixel_cnn, train_loader, test_loader, dict(epochs=25, lr=1e-3),
                                                      best_pixel_cnn)
    best_pixel_cnn.eval()
    images_samples = np.clip((best_pixel_cnn.sample() + 1) * 255 / 2, 0, 255)

    return np.array(train_losses), np.array(test_losses), np.array(train_pixel_cnn), np.array(test_pixel_cnn), np.floor(
        images_samples), np.floor(image_pairs)


def q3_():
    # q3_save_results(1, q3)
    q3_save_results(2, q3)


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, conditional_size=None,
                 color_conditioning=False, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.conditional_size = conditional_size
        self.color_conditioning = color_conditioning
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
            else:
                self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels,
                                         kernel_size=3, padding=1)

    def forward(self, input, cond=None):
        batch_size = input.shape[0]
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        if self.conditional_size:
            if len(self.conditional_size) == 1:
                # Broadcast across height and width of image and add as conditional bias
                out = out + self.cond_op(cond).view(batch_size, -1, 1, 1)
            else:
                out = out + self.cond_op(cond)
        return out

    def create_mask(self, mask_type):
        k = self.kernel_size[0]
        self.mask[:, :, :k // 2] = 1
        self.mask[:, :, k // 2, :k // 2] = 1
        if self.color_conditioning:
            assert self.in_channels % 3 == 0 and self.out_channels % 3 == 0
            one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
            if mask_type == 'B':
                self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[one_third_out:2 * one_third_out, :2 * one_third_in, k // 2, k // 2] = 1
                self.mask[2 * one_third_out:, :, k // 2, k // 2] = 1
            else:
                self.mask[one_third_out:2 * one_third_out, :one_third_in, k // 2, k // 2] = 1
                self.mask[2 * one_third_out:, :2 * one_third_in, k // 2, k // 2] = 1
        else:
            if mask_type == 'B':
                self.mask[:, :, k // 2, k // 2] = 1


class StackLayerNorm(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.h_layer_norm = LayerNorm(n_filters)
        self.v_layer_norm = LayerNorm(n_filters)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)
        vx, hx = self.v_layer_norm(vx), self.h_layer_norm(hx)
        return torch.cat((vx, hx), dim=1)


class GatedConv2d(nn.Module):
    def __init__(self, mask_type, in_channels, out_channels, k=7, padding=3):
        super().__init__()

        self.vertical = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=k,
                                  padding=padding, bias=False)
        self.horizontal = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, k),
                                    padding=(0, padding), bias=False)
        self.vtoh = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1,
                              bias=False)
        self.htoh = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                              bias=False)

        self.register_buffer('vmask', self.vertical.weight.data.clone())
        self.register_buffer('hmask', self.horizontal.weight.data.clone())

        self.vmask.fill_(1)
        self.hmask.fill_(1)

        # zero the bottom half rows of the vmask
        # No need for special color condition masking here since we get to see everything
        self.vmask[:, :, k // 2 + 1:, :] = 0

        # zero the right half of the hmask
        self.hmask[:, :, :, k // 2 + 1:] = 0
        if mask_type == 'A':
            self.hmask[:, :, :, k // 2] = 0

    def down_shift(self, x):
        x = x[:, :, :-1, :]
        pad = nn.ZeroPad2d((0, 0, 1, 0))
        return pad(x)

    def forward(self, x):
        vx, hx = x.chunk(2, dim=1)

        self.vertical.weight.data *= self.vmask
        self.horizontal.weight.data *= self.hmask

        vx = self.vertical(vx)
        hx_new = self.horizontal(hx)
        # Allow horizontal stack to see information from vertical stack
        hx_new = hx_new + self.vtoh(self.down_shift(vx))

        # Gates
        vx_1, vx_2 = vx.chunk(2, dim=1)
        vx = torch.tanh(vx_1) * torch.sigmoid(vx_2)

        hx_1, hx_2 = hx_new.chunk(2, dim=1)
        hx_new = torch.tanh(hx_1) * torch.sigmoid(hx_2)
        hx_new = self.htoh(hx_new)
        hx = hx + hx_new

        return torch.cat((vx, hx), dim=1)


# GatedPixelCNN using horizontal and vertical stacks to fix blind-spot
class GatedPixelCNN(nn.Module):
    def __init__(self, channel_in, channel_out, vq_vae, n_layers=4, n_filters=64):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criteria = nn.CrossEntropyLoss()
        self.n_channels = channel_in
        self.channel_out = channel_out
        self.latent_num = channel_out
        self.vae = vq_vae

        self.embedding = nn.Embedding(channel_out, self.n_channels)
        self.in_conv = MaskConv2d('A', self.n_channels, n_filters, 7, padding=3)
        model = []
        for _ in range(n_layers - 2):
            model.extend([nn.ReLU(), GatedConv2d('B', n_filters, n_filters, 7, padding=3)])
            model.append(StackLayerNorm(n_filters))
        self.out_conv = MaskConv2d('B', n_filters, self.channel_out, 7, padding=3)
        self.net = nn.Sequential(*model)

    def forward(self, x):
        x = x.to(self.device).float()
        self.vae.eval()
        with torch.no_grad():
            zq_index = self.vae(x)[-2]
        out = self.embedding(zq_index).permute(0, 3, 1, 2)
        out = self.in_conv(out)
        out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
        out = self.out_conv(out)
        return self.criteria(out, zq_index)

    def loss(self, x):
        return self(x)

    def sample(self):
        self.vae.eval()
        self.eval()
        with torch.no_grad():
            zq_index = torch.zeros((100, 8, 8)).to(self.device)
            for j in range(8):
                for k in range(8):
                    out = self.embedding(zq_index.long()).permute(0, 3, 1, 2)
                    out = self.in_conv(out)
                    out = self.net(torch.cat((out, out), dim=1)).chunk(2, dim=1)[1]
                    outs = self.out_conv(out)
                    outs = torch.softmax(outs, dim=1)
                    cate_distr = torch.distributions.Categorical(outs.permute(0, 2, 3, 1).reshape(-1, self.latent_num))
                    zq_index.data[:, j, k] = cate_distr.sample().reshape(outs.shape[0], 8, 8)[:, j, k]

            return self.vae.sample(zq_index).permute(0, 2, 3, 1).cpu().numpy()


def q4_a(train_data, test_data, dset_id):
    """
    train_data: An (n_train, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    test_data: An (n_test, 32, 32, 3) uint8 numpy array of color images with values in [0, 255]
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of VQ-VAE train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of VQ-VAE train losses evaluated once at initialization and after each epoch
    - a (# of training iterations,) numpy array of PixelCNN prior train losess evaluated every minibatch
    - a (# of epochs + 1,) numpy array of PixelCNN prior train losses evaluated once at initialization and after each epoch
    - a (100, 32, 32, 3) numpy array of 100 samples (an equal number from each class) with values in {0, ... 255}
    - a (100, 32, 32, 3) numpy array of 50 real image / reconstruction pairs
      FROM THE TEST SET with values in [0, 255]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    latents_dim = 256
    latents_num = 128
    model = VQVAE(latents_dim, latents_num).to(device)
    best_model = VQVAE(latents_dim, latents_num).to(device)
    pixel_cnn = GatedPixelCNN(64, latents_num, best_model).to(device)
    best_pixel_cnn = GatedPixelCNN(64, latents_num, best_model).to(device)

    train_data = (train_data / 255 - 0.5) * 2
    test_data = (test_data / 255 - 0.5) * 2
    images_set1 = train_data[0:50]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    train_losses, test_losses = vq_train_epochs(model, train_loader, test_loader, dict(epochs=30, lr=1e-3), best_model)
    best_model.eval()
    image_pairs = np.clip((best_model.sample_pairs(images_set1) + 1) * 255 / 2, 0, 255)

    train_pixel_cnn, test_pixel_cnn = vq_train_epochs(pixel_cnn, train_loader, test_loader, dict(epochs=30, lr=1e-3),
                                                      best_pixel_cnn)
    best_pixel_cnn.eval()
    images_samples = np.clip((best_pixel_cnn.sample() + 1) * 255 / 2, 0, 255)

    return np.array(train_losses), np.array(test_losses), np.array(train_pixel_cnn), np.array(test_pixel_cnn), np.floor(
        images_samples), np.floor(image_pairs)


def q4a():
    q4_a_save_results(2, q4_a)


def q4_b(train_data, test_data):
    """
    train_data: An (n_train, 28, 28, 1) uint8 numpy array of MNIST binary images
    test_data: An (n_test, 28, 28, 1) uint8 numpy array of MNIST binary images

    Returns
    - a (# of training iterations, 3) numpy array of full negative ELBO, reconstruction loss E[-log p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated every minibatch
    - a (# of epochs + 1, 3) numpy array of full negative ELBO, reconstruciton loss E[-p(x|z)],
      and KL term E[KL(q(z|x) | p(z))] evaluated once at initialization and after each epoch
    - a (100, 28, 28, 1) numpy array of 100 samples with values in {0, 1}
    - a (100, 28, 28, 1) numpy array of 50 real-image / reconstruction pairs with values in {0, 1}
    """

    """ YOUR CODE HERE """


def q4b():
    q4_b_save_results(q4_b)


if __name__ == '__main__':
    q4a()
