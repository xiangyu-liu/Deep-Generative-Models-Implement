from deepul.hw1_helper import *
import torch
import numpy as np


def update_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.9999


def q1_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """

    """ YOUR CODE HERE """
    theta = torch.zeros((d,), requires_grad=True, dtype=torch.float)
    batch_size = 40
    epoch = 60
    lr = 1e-2
    optimizer = torch.optim.Adam([theta, ], lr=lr)
    train_losses = []
    test_losses = []
    for iteration in range(epoch):
        for i in range(train_data.shape[0] // batch_size):
            batch_data = train_data[i * batch_size: (i + 1) * batch_size]
            loss = -torch.mean(
                torch.log(torch.exp(theta[batch_data]) / torch.sum(torch.exp(theta[list(np.arange(d))]))))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(torch.squeeze(loss))
        with torch.no_grad():
            evaluation = -torch.mean(
                torch.log(torch.exp(theta[test_data]) / torch.sum(torch.exp(theta[list(np.arange(d))]))))
            test_losses.append(evaluation)
        update_lr(optimizer)
    with torch.no_grad():
        pred = torch.exp(theta[list(np.arange(d))]) / torch.sum(torch.exp(theta[list(np.arange(d))]))
    return np.array(train_losses), np.array(test_losses), np.squeeze(pred.numpy())


def plot_q1_a():
    q1_save_results(1, 'a', q1_a)
    q1_save_results(2, 'a', q1_a)


def q1_b(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """

    """ YOUR CODE HERE """
    pi = torch.tensor([0.25, ] * 4, dtype=torch.float, requires_grad=True)
    ui = torch.tensor([0, 3, 6, 9], dtype=torch.float, requires_grad=True)
    si = torch.tensor([1, ] * 4, dtype=torch.float, requires_grad=True)
    batch_size = 40
    epoch = 100
    lr = 1e-2
    optimizer = torch.optim.Adam([pi, ui, si], lr=lr)
    train_losses = []
    test_losses = []
    test_data_tensor = torch.from_numpy(test_data)
    for iteration in range(epoch):
        for j in range(train_data.shape[0] // batch_size):
            batch_data = train_data[j * batch_size: (j + 1) * batch_size]
            batch_data_tensor = torch.from_numpy(batch_data)
            prob_per_item = sum([torch.exp(pi[i]) * (torch.sigmoid(
                (batch_data_tensor + 0.5 - ui[i]) / torch.exp(si[i]) + torch.from_numpy((batch_data == d - 1) * 1e6)) -
                                                     torch.sigmoid(
                                                         (batch_data_tensor - 0.5 - ui[i]) / torch.exp(
                                                             si[i]) - torch.from_numpy(
                                                             (batch_data == 0) * 1e6))) for i in range(4)]) / torch.sum(
                torch.exp(pi))
            loss = -torch.mean(torch.log(prob_per_item))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
        with torch.no_grad():
            prob_per_item = sum([torch.exp(pi[i]) * (torch.sigmoid(
                (test_data_tensor + 0.5 - ui[i]) / torch.exp(si[i]) + torch.from_numpy((test_data == d - 1) * 1e6)) -
                                                     torch.sigmoid(
                                                         (test_data_tensor - 0.5 - ui[i]) / torch.exp(
                                                             si[i]) - torch.from_numpy(
                                                             (test_data == 0) * 1e6))) for i in range(4)]) / torch.sum(
                torch.exp(pi))
            evaluation = -torch.mean(torch.log(prob_per_item))
        test_losses.append(evaluation)
    results_tensor = torch.tensor(list(range(d)))
    results_data = np.arange(d)
    with torch.no_grad():
        results = sum([torch.exp(pi[i]) * (torch.sigmoid(
            (results_tensor + 0.5 - ui[i]) / torch.exp(si[i]) + torch.from_numpy((results_data == d - 1) * 1e6)) -
                                           torch.sigmoid(
                                               (results_tensor - 0.5 - ui[i]) / torch.exp(
                                                   si[i]) - torch.from_numpy(
                                                   (results_data == 0) * 1e6))) for i in range(4)]) / torch.sum(
            torch.exp(pi))
    return train_losses, test_losses, results.numpy()


def plot_q1_b():
    q1_save_results(1, 'b', q1_b)
    q1_save_results(2, 'b', q1_b)


def visualize_q2a():
    visualize_q2a_data(dset_type=1)
    visualize_q2a_data(dset_type=2)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.sigmoid = torch.nn.Sigmoid()
        self.masked_matrix_list = masked_matrix_list

    def forward(self, inputs):
        inputs = inputs.to(device)
        # self.fc1.weight = torch.nn.Parameter(self.fc1.weight * self.masked_matrix_list[0].to(device),
        #                                      requires_grad=True)
        # self.fc2.weight = torch.nn.Parameter(self.fc2.weight * self.masked_matrix_list[1].to(device),
        #                                      requires_grad=True)
        # self.fc3.weight = torch.nn.Parameter(self.fc3.weight * self.masked_matrix_list[2].to(device),
        #                                      requires_grad=True)
        tmp = self.relu(self.fc1(inputs))
        tmp = self.relu(self.fc2(tmp))
        return self.sigmoid(self.fc3(tmp))


def q2_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """

    """ YOUR CODE HERE """
    np.random.seed(0)
    train_size = train_data.shape[0]
    variable_num = train_data.shape[1]
    test_size = test_data.shape[0]

    x1 = train_data[:, 0]
    x1_one_hot = np.zeros((train_size, d), dtype=np.int8)
    x1_one_hot[np.arange(train_size), x1] = 1
    x2 = train_data[:, 1]
    x2_one_hot = np.zeros((train_size, d), dtype=np.int8)
    x2_one_hot[np.arange(train_size), x2] = 1

    test_x1 = test_data[:, 0]
    test_x1_one_hot = np.zeros((test_size, d), dtype=np.int8)
    test_x1_one_hot[np.arange(test_size), test_x1] = 1
    test_x2 = test_data[:, 1]
    test_x2_one_hot = np.zeros((test_size, d), dtype=np.int8)
    test_x2_one_hot[np.arange(test_size), test_x2] = 1

    hidden_units = [variable_num * d, 400, 400, variable_num * d]
    m0 = np.array([1, ] * d + [2, ] * d, dtype=np.int8)
    m1 = np.random.randint(low=1, high=3, size=hidden_units[1])
    m2 = np.random.randint(low=1, high=3, size=hidden_units[2])
    m3 = np.array([1, ] * d + [2, ] * d, dtype=np.int8)
    M1 = torch.from_numpy(m1[np.newaxis].transpose() >= m0[np.newaxis])
    M2 = torch.from_numpy(m2[np.newaxis].transpose() >= m1[np.newaxis])
    M3 = torch.from_numpy(m3[np.newaxis].transpose() > m2[np.newaxis])
    masked_matrices = [M1, M2, M3]
    model = Made(masked_matrices, hidden_units).to(device)

    batch_size = 4000
    epoch = 50
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    for iteration in range(epoch):
        for i in range(train_size // batch_size):
            batch_data = np.concatenate(
                (x1_one_hot[i * batch_size: (i + 1) * batch_size], x2_one_hot[i * batch_size: (i + 1) * batch_size]),
                axis=1)
            outs = model(torch.from_numpy(batch_data).type(torch.float))
            out1, out2 = torch.softmax(outs[:, 0:d], dim=1), torch.softmax(outs[:, d:(2 * d)], dim=1)
            prob1 = out1[np.arange(batch_size), x1[i * batch_size: (i + 1) * batch_size]]
            prob2 = out2[np.arange(batch_size), x2[i * batch_size: (i + 1) * batch_size]]
            loss = -torch.mean(torch.log(prob1) + torch.log(prob2))/2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
        with torch.no_grad():
            test_data_tensor = torch.from_numpy(np.concatenate((test_x1_one_hot, test_x2_one_hot), axis=1)).type(
                torch.float)
            outs = model(test_data_tensor)
            out1, out2 = torch.softmax(outs[:, 0:d], dim=1), torch.softmax(outs[:, d:(2 * d)], dim=1)
            prob1, prob2 = out1[np.arange(test_size), test_x1], out2[np.arange(test_size), test_x2]
            test_loss = -torch.mean(torch.log(prob1) + torch.log(prob2))/2
            test_losses.append(test_loss)
            if iteration % 10 == 0:
                print("\niteration {}\ntraining losses {}\nevaluation losses {}".format(iteration, loss, test_loss))
    with torch.no_grad():
        x1_col = np.ravel(np.array([i for i in range(d)]))
        x1_col_one_hot = np.zeros((d, d), dtype=np.int8)
        x1_col_one_hot[np.arange(d), x1_col] = 1

        x2_col = np.array(list(range(d)))
        x2_col_one_hot = np.zeros((d, d), dtype=np.int8)
        x2_col_one_hot[np.arange(d), x2_col] = 1

        final_data = np.concatenate((x1_col_one_hot, x2_col_one_hot), axis=1)
        outs = model(torch.from_numpy(final_data).type(torch.float))
        px1 = torch.softmax(outs[0, :d], dim=0).view(-1, 1)
        px2_condition_on1 = torch.softmax(outs[:, d:(2 * d)], dim=1)
        result = px1 * px2_condition_on1
    return train_losses, test_losses, result.cpu().numpy()


def plot_q2_a():
    q2_save_results(1, 'a', q2_a)
    q2_save_results(2, 'a', q2_a)


def visualize_q2_b():
    visualize_q2b_data(1)
    visualize_q2b_data(2)


def q2_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """
    np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    train_size = train_data.shape[0]
    variable_num = train_data.shape[1]
    test_size = test_data.shape[0]

    hidden_units = [variable_num, 800, 800, variable_num]
    m0 = np.array([i + 1 for i in range(variable_num)], dtype=np.int16)
    m1 = np.random.randint(low=1, high=variable_num + 1, size=hidden_units[1])
    m2 = np.random.randint(low=1, high=variable_num + 1, size=hidden_units[2])
    m3 = np.array([i + 1 for i in range(variable_num)], dtype=np.int16)
    M1 = torch.from_numpy(m1[np.newaxis].transpose() >= m0[np.newaxis])
    M2 = torch.from_numpy(m2[np.newaxis].transpose() >= m1[np.newaxis])
    M3 = torch.from_numpy(m3[np.newaxis].transpose() > m2[np.newaxis])
    masked_matrices = [M1, M2, M3]
    model = Made(masked_matrices, hidden_units).to(device)

    batch_size = 4000
    epoch = 50
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    criteria = torch.nn.BCELoss()
    for iteration in range(epoch):
        for i in range(train_size // batch_size):
            batch_data = train_data[i * batch_size: (i + 1) * batch_size]
            outs = model(torch.from_numpy(batch_data).type(torch.float))
            loss = criteria(outs, torch.from_numpy(batch_data).type(torch.float).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
        with torch.no_grad():
            batch_data = test_data
            outs = model(torch.from_numpy(batch_data).type(torch.float))
            test_loss = criteria(outs, torch.from_numpy(batch_data).type(torch.float).to(device))
            test_losses.append(test_loss)
            if iteration % 10 == 0:
                print("\niteration {}\ntraining losses {}\nevaluation losses {}".format(iteration, loss, test_loss))
    with torch.no_grad():
        image_list = []
        for i in range(100):
            pixel_list = []
            for j in range(variable_num):
                outs = model(torch.tensor(pixel_list + [0, ] * (variable_num - len(pixel_list)), dtype=torch.float))
                px = outs[j]
                pixel_list.append(1 if np.random.random() < px else 0)
            image_list.append(np.array(pixel_list, dtype=np.int8).reshape(image_shape))
    return train_losses, test_losses, np.stack(image_list, axis=0).reshape([100, ] + list(image_shape) + [1, ])


# ### Results
#
# Once you've implemented `q2_b`, execute the cells below to visualize and save your results
#
#
def plot_q2_b():
    q2_save_results(1, 'b', q2_b)
    q2_save_results(2, 'b', q2_b)


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
            MaskedConv2d('A', channel_in, 64, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('A', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('A', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
        )

        self.maskB_block = torch.nn.Sequential(
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, 1, 3),
            torch.nn.ReLU()

        )

        self.one_one_conv = torch.nn.Sequential(
            MaskedConv2d('B', 64, 32, 1, 1, 0),
            torch.nn.ReLU(),
            MaskedConv2d('B', 32, channel_out, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = inputs.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        outs = self.maskA_block(inputs)
        outs = self.maskB_block(outs)
        return self.one_one_conv(outs)


def q3_a(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: A (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    """ YOUR CODE HERE """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PixelCNN(1, 1).to(device)
    epoch = 30
    lr = 1e-3
    batch_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    criteria = torch.nn.BCELoss()
    for iteration in range(epoch):
        for i in range(train_data.shape[0] // batch_size):
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]
            batch_data_tensor = torch.from_numpy(batch_data).permute(0, 3, 1, 2).type(torch.float)
            outs = model(2 * (batch_data_tensor - 0.5))
            loss = criteria(outs, batch_data_tensor.to(device))
            # loss = torch.mean(torch.mul(outs - batch_data_tensor.to(device), outs - batch_data_tensor.to(device)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            test_data_tensor = torch.from_numpy(test_data).permute(0, 3, 1, 2).type(torch.float)
            outs = model(2 * (test_data_tensor - 0.5))
            evaluation = criteria(outs, test_data_tensor.to(device))
            train_losses.append(loss)
            test_losses.append(evaluation)
            print("\niteration {}\ntrain loss {}\ntest loss {}".format(iteration, loss, evaluation))
    with torch.no_grad():
        image = torch.zeros((100, image_shape[0], image_shape[1], 1))
        for j in range(image_shape[0]):
            for k in range(image_shape[1]):
                outs = model(2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5))
                image.data[:, j, k, 0] = torch.from_numpy(
                    np.random.random(size=(100,)) < outs[:, 0, j, k].cpu().numpy())
    return train_losses, test_losses, image.numpy()


def plot_q3_a():
    q3a_save_results(1, q3_a)
    q3a_save_results(2, q3_a)


def visualize_q3_b():
    visualize_q3b_data(1)
    visualize_q3b_data(2)


class MaskConvBlock(torch.nn.Module):
    def __init__(self):
        super(MaskConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            MaskedConv2d('B', 120, 120, 1, 1, 0),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(120),
            MaskedConv2d('B', 120, 120, 7, 1, 3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(120),
            MaskedConv2d('B', 120, 120, 1, 1, 0)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.block(inputs) + inputs)


class ColoredPixelCNN(torch.nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.maskA_block = torch.nn.Sequential(
            MaskedConv2d('A', channel_in, 120, 7, 1, 3),
            torch.nn.ReLU(),
            MaskedConv2d('A', 120, 120, 7, 1, 3),
            torch.nn.ReLU(),
        )
        self.maskB_block = torch.nn.Sequential(*([MaskConvBlock()] * 8))
        self.one_one_conv = torch.nn.Sequential(
            MaskedConv2d('B', 120, 60, 1, 1, 0),
            torch.nn.ReLU(),
            MaskedConv2d('B', 60, 30, 1, 1, 0),
            torch.nn.ReLU(),
            MaskedConv2d('B', 30, channel_out, 1, 1, 0)
        )

    def forward(self, inputs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        outs = self.maskA_block(inputs)
        outs = self.maskB_block(outs)
        return self.one_one_conv(outs)


def q3_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3}
    """

    """ YOUR CODE HERE """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ColoredPixelCNN(train_data.shape[-1], train_data.shape[-1] * 4).to(device)
    epoch = 16
    lr = 1e-3
    batch_size = 16
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    criteria = torch.nn.CrossEntropyLoss()
    best_loss = 9999
    best_model = ColoredPixelCNN(train_data.shape[-1], train_data.shape[-1] * 4).to(device)
    for iteration in range(epoch):
        np.random.shuffle(train_data)
        for i in range(train_data.shape[0] // batch_size):
            model.train()
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]
            batch_data_tensor = torch.from_numpy(batch_data).permute(0, 3, 1, 2).to(device)
            outs = model(2 * (batch_data_tensor.type(torch.float) - 0.5))
            outs_ch1 = outs[:, 0:4]
            outs_ch2 = outs[:, 4:8]
            outs_ch3 = outs[:, 8:12]
            # loss = criteria(outs_ch1, batch_data_tensor[:, 0].type(torch.long)) \
            #        + criteria(outs_ch2, batch_data_tensor[:, 1].type(torch.long)) \
            #        + criteria(outs_ch3, batch_data_tensor[:, 2].type(torch.long))
            # loss2 = criteria(outs.permute(0, 2, 3, 1).reshape(batch_size, image_shape[0], image_shape[1], 3, 4).permute(0, -1, -2, 1, 2), batch_data_tensor.type(torch.long).to(device))
            loss = (criteria(outs_ch1.permute(0, 2, 3, 1).reshape(-1, outs_ch1.shape[1]),
                            torch.squeeze(batch_data_tensor.permute(0, 2, 3, 1)[:, :, :, 0].reshape(-1, 1)).type(
                                torch.long)) + \
                   criteria(outs_ch2.permute(0, 2, 3, 1).reshape(-1, outs_ch2.shape[1]),
                            torch.squeeze(batch_data_tensor.permute(0, 2, 3, 1)[:, :, :, 1].reshape(-1, 1)).type(
                                torch.long)) + \
                   criteria(outs_ch3.permute(0, 2, 3, 1).reshape(-1, outs_ch3.shape[1]),
                            torch.squeeze(batch_data_tensor.permute(0, 2, 3, 1)[:, :, :, 2].reshape(-1, 1)).type(
                                torch.long)))

            # outs_ch1 = torch.softmax(outs[:, 0:4], dim=1).permute(0, 2, 3, 1).reshape(-1, 4)
            # outs_ch2 = torch.softmax(outs[:, 4:8], dim=1).permute(0, 2, 3, 1).reshape(-1, 4)
            # outs_ch3 = torch.softmax(outs[:, 8:12], dim=1).permute(0, 2, 3, 1).reshape(-1, 4)
            # index1 = np.ravel(batch_data[:, :, :, 0]).astype(np.int)
            # index2 = np.ravel(batch_data[:, :, :, 1]).astype(np.int)
            # index3 = np.ravel(batch_data[:, :, :, 2]).astype(np.int)
            # loss_ = -(torch.mean(torch.log(outs_ch1[np.arange(batch_size*image_shape[0]*image_shape[1]), index1])) +
            #         torch.mean(torch.log(outs_ch2[np.arange(batch_size*image_shape[0]*image_shape[1]), index2])) +
            #         torch.mean(torch.log(outs_ch3[np.arange(batch_size * image_shape[0] * image_shape[1]), index3])))/3
            # print(loss/3, loss_, loss2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
        with torch.no_grad():
            model.eval()
            batch_data = test_data[0:1000]
            batch_data_tensor = torch.from_numpy(batch_data).permute(0, 3, 1, 2).to(device)
            outs = model(2 * (batch_data_tensor.type(torch.float) - 0.5))
            outs_ch1 = outs[:, 0:4]
            outs_ch2 = outs[:, 4:8]
            outs_ch3 = outs[:, 8:12]
            evaluation = criteria(outs_ch1, batch_data_tensor[:, 0].type(torch.long)) \
                         + criteria(outs_ch2, batch_data_tensor[:, 1].type(torch.long)) \
                         + criteria(outs_ch3, batch_data_tensor[:, 2].type(torch.long))
            # evaluation = criteria(outs.permute(0, 2, 3, 1).reshape(test_data.shape[0], image_shape[0], image_shape[1], 3, 4).permute(0, -1, -2, 1, 2), batch_data_tensor.type(torch.long).to(device))
            test_losses.append(evaluation / 3)
            print("\niteration {}\ntrain loss {}\ntest loss {}".format(iteration, loss, evaluation / 3))
            if evaluation / 3 < best_loss:
                best_loss = evaluation / 3
                best_model.load_state_dict(model.state_dict())
    model = best_model
    print("\nbest loss is {}".format(best_loss))
    with torch.no_grad():
        model.eval()
        image = torch.zeros((100, image_shape[0], image_shape[1], train_data.shape[-1])).to(device)
        for j in range(image_shape[0]):
            for k in range(image_shape[1]):
                outs = model(2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5))
                outs_ch1 = torch.softmax(outs[:, 0:4], dim=1)
                image.data[:, j, k, 0] = torch.distributions.Categorical(
                    outs_ch1.permute(0, 2, 3, 1).reshape(-1, outs_ch1.shape[1])).sample().reshape(outs_ch1.shape[0],
                                                                                                  outs_ch1.shape[2],
                                                                                                  outs_ch1.shape[3])[:,
                                         j, k]
                outs_ch2 = torch.softmax(outs[:, 4:8], dim=1)
                image.data[:, j, k, 1] = torch.distributions.Categorical(
                    outs_ch2.permute(0, 2, 3, 1).reshape(-1, outs_ch2.shape[1])).sample().reshape(outs_ch2.shape[0],
                                                                                                  outs_ch2.shape[2],
                                                                                                  outs_ch2.shape[3])[:,
                                         j, k]
                outs_ch3 = torch.softmax(outs[:, 8:12], dim=1)
                image.data[:, j, k, 2] = torch.distributions.Categorical(
                    outs_ch3.permute(0, 2, 3, 1).reshape(-1, outs_ch3.shape[1])).sample().reshape(outs_ch3.shape[0],
                                                                                                  outs_ch3.shape[2],
                                                                                                  outs_ch3.shape[3])[:,
                                         j, k]

    return train_losses, test_losses, image.cpu().numpy()
    # with torch.no_grad():
    #     image_list = []
    #     model.eval()
    #     for i in range(100):
    #         image = torch.zeros((1, image_shape[0], image_shape[1], train_data.shape[-1]))
    #         for j in range(image_shape[0]):
    #             for k in range(image_shape[1]):
    #                 outs = model(2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5))

    #                 outs_ch1 = torch.softmax(outs[0, 0:4], dim=0)
    #                 image.data[0, j, k, 0] = torch.distributions.Categorical(
    #                     outs_ch1.permute(1, 2, 0).reshape(-1, outs_ch1.shape[0])).sample().reshape(image_shape[0],image_shape[1])[j, k]
    #                 outs_ch2 = torch.softmax(outs[0, 4:8], dim=0)
    #                 image.data[0, j, k, 1] = torch.distributions.Categorical(
    #                     outs_ch2.permute(1, 2, 0).reshape(-1, outs_ch2.shape[0])).sample().reshape(image_shape[0],image_shape[1])[j, k]
    #                 outs_ch3 = torch.softmax(outs[0, 8:12], dim=0)
    #                 image.data[:, j, k, 2] = torch.distributions.Categorical(
    #                     outs_ch3.permute(1, 2, 0).reshape(-1, outs_ch3.shape[0])).sample().reshape(image_shape[0],image_shape[1])[j, k]
    #         image_list.append(image.numpy())
    # return train_losses, test_losses, np.concatenate(image_list, axis=0)


def plot_q3b():
    # q3bc_save_results(1, 'b', q3_b)
    q3bc_save_results(2, 'b', q3_b)


class AugmentMaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad):
        """2D Convolution with masked weight for Autoregressive connection"""
        super(AugmentMaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias=False)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()
        num_per_group = ch_out // 3
        one_third_channels = ch_in // 3
        mask = torch.ones(ch_out, ch_in, height, width)
        # define the first group mask
        mask[0:num_per_group, :, height // 2, width // 2:] = 0
        mask[0:num_per_group, :, height // 2 + 1:, :] = 0
        mask[0:num_per_group, 0:one_third_channels, height // 2, width // 2] = 1 if self.mask_type == 'B' else 0
        # define the second group mask
        mask[num_per_group:2 * num_per_group, :, height // 2, width // 2:] = 0
        mask[num_per_group:2 * num_per_group, :, height // 2 + 1:, :] = 0
        mask[num_per_group:2 * num_per_group, 0:one_third_channels, height // 2, width // 2] = 1
        mask[num_per_group:2 * num_per_group, one_third_channels:2 * one_third_channels, height // 2,
        width // 2] = 1 if self.mask_type == 'B' else 0
        # define the third group mask
        mask[2 * num_per_group:, :, height // 2, width // 2:] = 0
        mask[2 * num_per_group:, :, height // 2 + 1:, :] = 0
        mask[2 * num_per_group:, 0:2 * one_third_channels, height // 2, width // 2] = 1
        mask[2 * num_per_group:, 2 * one_third_channels:, height // 2, width // 2] = 1 if self.mask_type == 'B' else 0
        self.register_buffer('mask', mask)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(AugmentMaskedConv2d, self).forward(x)


class AugmentMaskConvBlock(torch.nn.Module):
    def __init__(self):
        super(AugmentMaskConvBlock, self).__init__()
        self.block = torch.nn.Sequential(
            AugmentMaskedConv2d('B', 120, 120, 1, 1, 0),
            torch.nn.ReLU(),
            AugmentMaskedConv2d('B', 120, 120, 7, 1, 3),
            torch.nn.ReLU(),
            AugmentMaskedConv2d('B', 120, 120, 1, 1, 0)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.block(inputs) + inputs)


class AugmentColoredPixelCNN(torch.nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.maskA_block = torch.nn.Sequential(
            AugmentMaskedConv2d('A', channel_in, 120, 7, 1, 3),
            torch.nn.ReLU(),
            AugmentMaskedConv2d('A', 120, 120, 7, 1, 3),
            torch.nn.ReLU(),
        )
        self.maskB_block = torch.nn.Sequential(*([AugmentMaskConvBlock()] * 8))
        self.one_one_conv = torch.nn.Sequential(
            AugmentMaskedConv2d('B', 120, 60, 1, 1, 0),
            torch.nn.ReLU(),
            AugmentMaskedConv2d('B', 60, channel_out, 1, 1, 0)
        )

    def forward(self, inputs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)
        outs = self.maskA_block(inputs)
        outs = self.maskB_block(outs)
        return self.one_one_conv(outs)


def q3_c(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, W) of samples with values in {0, 1, 2, 3}
    """

    """ YOUR CODE HERE """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AugmentColoredPixelCNN(train_data.shape[-1], train_data.shape[-1] * 4).to(device)
    epoch = 30
    lr = 1e-3
    batch_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    criteria = torch.nn.CrossEntropyLoss()
    best_loss = 9999
    best_model = AugmentColoredPixelCNN(train_data.shape[-1], train_data.shape[-1] * 4).to(device)
    for iteration in range(epoch):
        for i in range(train_data.shape[0] // batch_size):
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]
            batch_data_tensor = torch.from_numpy(batch_data).permute(0, 3, 1, 2).to(device)
            outs = model(2 * (batch_data_tensor.type(torch.float) - 0.5))
            outs_ch1 = outs[:, 0:4]
            outs_ch2 = outs[:, 4:8]
            outs_ch3 = outs[:, 8:12]
            # loss = criteria(outs_ch1, batch_data_tensor[:, 0].type(torch.long)) \
            #        + criteria(outs_ch2, batch_data_tensor[:, 1].type(torch.long)) \
            #        + criteria(outs_ch3, batch_data_tensor[:, 2].type(torch.long))
            # loss = criteria(
            #     torch.softmax(outs.permute(0, 2, 3, 1).reshape(batch_size, image_shape[0], image_shape[1], 3, 4),
            #                   dim=-1).permute(0, -1, -2, 1, 2), batch_data_tensor.type(torch.long).to(device))
            loss = criteria(outs_ch1.permute(0, 2, 3, 1).reshape(-1, outs_ch1.shape[1]),
                            torch.squeeze(batch_data_tensor.permute(0, 2, 3, 1)[:, :, :, 0].reshape(-1, 1)).type(
                                torch.long)) + \
                   criteria(outs_ch2.permute(0, 2, 3, 1).reshape(-1, outs_ch2.shape[1]),
                            torch.squeeze(batch_data_tensor.permute(0, 2, 3, 1)[:, :, :, 1].reshape(-1, 1)).type(
                                torch.long)) + \
                   criteria(outs_ch3.permute(0, 2, 3, 1).reshape(-1, outs_ch3.shape[1]),
                            torch.squeeze(batch_data_tensor.permute(0, 2, 3, 1)[:, :, :, 2].reshape(-1, 1)).type(
                                torch.long))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            batch_data = test_data
            batch_data_tensors = torch.from_numpy(batch_data).permute(0, 3, 1, 2).to(device)
            outs = model(2 * (batch_data_tensors.type(torch.float) - 0.5))
            outs_ch1 = outs[:, 0:4]
            outs_ch2 = outs[:, 4:8]
            outs_ch3 = outs[:, 8:12]
            evaluation = criteria(outs_ch1, batch_data_tensors[:, 0].type(torch.long)) \
                         + criteria(outs_ch2, batch_data_tensors[:, 1].type(torch.long)) \
                         + criteria(outs_ch3, batch_data_tensors[:, 2].type(torch.long))
            train_losses.append(loss/3)
            test_losses.append(evaluation/3)
            print("\niteration {}\ntrain loss {}\ntest loss {}".format(iteration, loss / 3, evaluation / 3))
            if evaluation/3 < best_loss:
              best_loss = evaluation/3
              best_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        image = torch.zeros((100, image_shape[0], image_shape[1], train_data.shape[-1])).to(device)
        for j in range(image_shape[0]):
            for k in range(image_shape[1]):
                outs = model(2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5))
                outs_ch1 = torch.softmax(outs[:, 0:4], dim=1)
                image.data[:, j, k, 0] = torch.distributions.Categorical(
                    outs_ch1.permute(0, 2, 3, 1).reshape(-1, outs_ch1.shape[1])).sample().reshape(outs_ch1.shape[0],outs_ch1.shape[2],outs_ch1.shape[3])[:,j, k]
                outs_ch2 = torch.softmax(model(2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5))[:, 4:8], dim=1)
                image.data[:, j, k, 1] = torch.distributions.Categorical(
                    outs_ch2.permute(0, 2, 3, 1).reshape(-1, outs_ch2.shape[1])).sample().reshape(outs_ch2.shape[0],outs_ch2.shape[2],outs_ch2.shape[3])[:,j, k]
                outs_ch3 = torch.softmax(model(2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5))[:, 8:12], dim=1)
                image.data[:, j, k, 2] = torch.distributions.Categorical(
                    outs_ch3.permute(0, 2, 3, 1).reshape(-1, outs_ch3.shape[1])).sample().reshape(outs_ch3.shape[0],outs_ch3.shape[2],outs_ch3.shape[3])[:,j, k]
    return train_losses, test_losses, image.cpu().numpy()


def plot_q3c():
    q3bc_save_results(1, 'c', q3_c)
    q3bc_save_results(2, 'c', q3_c)


class InfoMaskedConv2d(torch.nn.Conv2d):
    def __init__(self, mask_type, c_in, c_out, k_size, stride, pad, num_classes, use_relu=True):
        """2D Convolution with masked weight for Autoregressive connection"""
        super(InfoMaskedConv2d, self).__init__(
            c_in, c_out, k_size, stride, pad, bias=False)
        assert mask_type in ['A', 'B']
        self.mask_type = mask_type
        ch_out, ch_in, height, width = self.weight.size()

        mask = torch.ones(ch_out, ch_in, height, width)
        mask[:, :, height // 2, width // 2:] = 0
        mask[:, :, height // 2 + 1:, :] = 0
        mask[:, :, height // 2, width // 2] = 1 if self.mask_type == 'B' else 0
        self.register_buffer('mask', mask)

        self.bias_weight = torch.nn.Parameter(torch.ones(ch_out, num_classes, requires_grad=True), requires_grad=True)
        self.relu = torch.nn.ReLU()
        self.use_relu = use_relu

    def forward(self, x):
        self.weight.data *= self.mask
        tmp = torch.unsqueeze(torch.unsqueeze(torch.mm(x[1].type(torch.float),self.bias_weight.T), dim=-1), dim=-1)
        if self.use_relu:
            return [self.relu(super(InfoMaskedConv2d, self).forward(x[0]) + tmp), x[1]]
        else:
            return [super(InfoMaskedConv2d, self).forward(x[0]) + tmp, x[1]]


class InfoPixelCNN(torch.nn.Module):
    def __init__(self, channel_in, channel_out, num_classes):
        super().__init__()
        self.maskA_block = torch.nn.Sequential(
            InfoMaskedConv2d('A', channel_in, 64, 7, 1, 3, num_classes),
            InfoMaskedConv2d('A', 64, 64, 7, 1, 3, num_classes),
            InfoMaskedConv2d('A', 64, 64, 7, 1, 3, num_classes),
        )

        self.maskB_block = torch.nn.Sequential(
            InfoMaskedConv2d('B', 64, 64, 7, 1, 3, num_classes),
            InfoMaskedConv2d('B', 64, 64, 7, 1, 3, num_classes),
            InfoMaskedConv2d('B', 64, 64, 7, 1, 3, num_classes),
            InfoMaskedConv2d('B', 64, 64, 7, 1, 3, num_classes),
            InfoMaskedConv2d('B', 64, 64, 7, 1, 3, num_classes),
        )

        self.one_one_conv = torch.nn.Sequential(
            InfoMaskedConv2d('B', 64, 32, 1, 1, 0, num_classes),
            InfoMaskedConv2d('B', 32, channel_out, 1, 1, 0, num_classes, use_relu=False),
        )
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, inputs):
        inputs[0] = inputs[0].to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        inputs[1] = inputs[1].to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        outs = self.maskA_block(inputs)
        outs = self.maskB_block(outs)
        outs = self.one_one_conv(outs)[0]
        return self.sigmoid(outs)


def q3_d(train_data, train_labels, test_data, test_labels, image_shape, n_classes, dset_id):
    """
    train_data: A (n_train, H, W, 1) numpy array of binary images with values in {0, 1}
    train_labels: A (n_train,) numpy array of class labels
    test_data: A (n_test, H, W, 1) numpy array of binary images with values in {0, 1}
    test_labels: A (n_test,) numpy array of class labels
    image_shape: (H, W), height and width
    n_classes: number of classes (4 or 10)
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, C, 1) of samples with values in {0, 1}
      where an even number of images of each class are sampled with 100 total
    """

    """ YOUR CODE HERE """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = InfoPixelCNN(1, 1, n_classes).to(device)
    epoch = 1
    lr = 1e-3
    batch_size = 128
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    criteria = torch.nn.BCELoss()
    best_loss = 9999
    best_model = InfoPixelCNN(1, 1, n_classes).to(device)
    for iteration in range(epoch):
        for i in range(train_data.shape[0] // batch_size):
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]
            batch_label = train_labels[i * batch_size:(i + 1) * batch_size]
            batch_label_one_hot = np.zeros((batch_size, n_classes), dtype=np.int8)
            batch_label_one_hot[np.arange(batch_size), batch_label] = 1
            batch_label_one_hot = torch.from_numpy(batch_label_one_hot)
            batch_data_tensor = torch.from_numpy(batch_data).permute(0, 3, 1, 2).type(torch.float)
            outs = model([2 * (batch_data_tensor - 0.5), batch_label_one_hot])
            loss = criteria(outs, batch_data_tensor.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss)
            print("minibatch {}\nloss {}\n".format(i, loss))
        # with torch.no_grad():
        #     batch_data = test_data
        #     batch_label = test_labels
        #     batch_label_one_hot = np.zeros((test_data.shape[0], n_classes), dtype=np.int8)
        #     batch_label_one_hot[np.arange(test_data.shape[0]), batch_label] = 1
        #     batch_label_one_hot = torch.from_numpy(batch_label_one_hot)
        #     batch_data_tensor = torch.from_numpy(batch_data).permute(0, 3, 1, 2).type(torch.float)
        #     outs = model([2 * (batch_data_tensor - 0.5), batch_label_one_hot])
        #     evaluation = criteria(outs, batch_data_tensor.to(device))
        #     test_losses.append(evaluation)
        #     print("\niteration {}\ntrain loss {}\ntest loss {}".format(iteration, loss, evaluation))
    #         if evaluation<best_loss:
    #             best_loss = evaluation
    #             best_model.load_state_dict(model.state_dict())
    # model = best_model
    with torch.no_grad():
        image = torch.zeros((100, image_shape[0], image_shape[1], 1))
        batch_label = []
        for _ in range(n_classes):
            batch_label = batch_label + [_]*(100//n_classes)
        batch_label_one_hot = np.zeros((100, n_classes), dtype=np.int8)
        batch_label_one_hot[np.arange(100), batch_label] = 1
        batch_label_one_hot = torch.from_numpy(batch_label_one_hot)
        for j in range(image_shape[0]):
            for k in range(image_shape[1]):
                outs = model([2 * (image.permute(0, 3, 1, 2).type(torch.float) - 0.5), batch_label_one_hot])
                image.data[:, j, k, 0] = torch.from_numpy(
                    np.random.random(size=(100,)) < outs[:, 0, j, k].cpu().numpy())
    return train_losses, test_losses, image.numpy()


def plot_q3d():
    q3d_save_results(1, q3_d)
    q3d_save_results(2, q3_d)


def q4_a(train_data, test_data, image_shape):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, C) of generated samples with values in {0, 1, 2, 3}
    """
    """ YOUR CODE HERE """


def plot_q4a():
    q4a_save_results(q4_a)


def q4_b(train_data, test_data, image_shape):
    """
    train_data: A (n_train, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    test_data: A (n_test, H, W, C) uint8 numpy array of color images with values in {0, 1, 2, 3}
    image_shape: (H, W, C), height, width, and # of channels of the image

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (50, H, W, 1) of generated binary images in {0, 1}
    - a numpy array of size (50, H, W, C) of conditonally generated color images in {0, 1, 2, 3}
    """
    # You will need to generate the binary image dataset from train_data and test_data

    """ YOUR CODE HERE """


def plot_q4b():
    q4b_save_results(q4_b)


def q4_c(train_data, test_data):
    """
    train_data: A (60000, 56, 56, 1) numpy array of grayscale images with values in {0, 1}
    test_data: A (10000, 56, 56, 1) numpy array of grayscale images with values in {0, 1}
    image_shape: (H, W), height and width

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, 56, 56, 1) of generated samples with values in {0, 1}
    """

    """ YOUR CODE HERE """


def plot_q4c():
    q4c_save_results(q4_c)


if __name__ == '__main__':
    index = np.array([[0, 1], [1, 0]])
    matrix = np.random.random(size=(2, 2, 2))
    plot_q3b()

'''
Some notes:

'''