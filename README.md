# CS 294-158 Deep Unsupervised Learning

The official public repository for the Spring 2020 offering of Berkeley's CS294-158 Deep Unsupervised Learning.

## Some Pytorch Notes

### From HW1

- **torch.repeat()**: to repeat the tensor according to a specific axis. If one element is needed by four function, then we can repeat it by several times along one specific axis
- **torch.where()**: according to a bool expression to choose element from two given arrays (choose one from two)
- **torch.gather()**: choose elements from one specific axis. eg: MLE true label can serve as the index. shape[0] is the batch size
- **torch.nn.ModuleList()**: This will help us register the parameters and will be useful when we need parameter sharing between different layers and store the temporary results
- **torch.reshape(3, 4) and torch.reshape(4, 3)** can be very very different
- **torch.nn.LayerNorm**: by default normalize the final dimension
- **torch.nn.CrossEntropy()**: It will do softmax for you. So we must not softmax the logits

### From HW2

- **torch.normal(mu, std)** will directly return the random number, shape of which is identical to the shape of mu and std
- **torch.chunk(tensor, n_split, dim)** can be useful when we split channels into n_split groups
- **Normal().log_prob()**, where normal can receive the tensors as inputs

### Some lessons from hw2

- **Pseudo codes can be extremely useful**, especially when the model is complicated. This can increase coding efficiency and decrease potential bugs!
- **The last layer of the neural network can be critical**. Sometimes, large network is not that expressive is due to the wrong choices of the last layer, (but we can also fix it by using the idea from weight normalization)
- **For PixelCNN, we do not need many type A mask and one-to-one convolution**. Instead, increase the type B mask can increase the representation ability of the networks.
- **The dequantization is not fixed at the beginning. Instead, each time we take a epoch from the dataloader, we add random noise**. This can be viewed as regularization for deep NN
- **Sometimes the outputs for NN can become nan**. One potential reason is that the learning rate is too large. A quite useful trick for generative model is to increase the lr linearly from 0 to a fixed value (eg: 5e-4) during the first (200) minibatches.
- **The idea of weight normalization can be helpful.** In the coupling layer, the solution further apply **tanh()** to **log_s** and the multiply by **self.scale** and then add by **self.shift**. scale and shift are learnable parameters
- **The logit function is log(x/(1-x)) and the logit trick is wrong in the original realnvp paper**
- 

### Some lessons from hw3

- **nn.embeddings can be useful:** For VQ-VAE, the input for the the prior PixelCNN is the integer. We need to first convert it to embeddings (numerical vectors) and then apply the convolutional layers. embedding layer is a trainable look-up table, mapping the index to embeddings
- **The blur of images in VAE is caused by the continuity of the latents:** Increasing the latents dimension can cause blur of images
- **Cuda memory issues: It may be caused by storing loss with gradients**. When recoding the loss, use loss.cpu.item()
- **Batch normalization can be quite useful**



### Some lessons from hw4

- **When training GAN, discriminators sometimes may need more training than generators (n_critic usually range from 1 to 5)**
- **When training GAN, hyperparameters of Adam also need tuning**
- **When training G and D separately, remember to call .detach()**
- **The final layer of D depends on the model:** Sometime we use Tanh(), but sometime we do not use activation. It depends
- **We sometimes use leakyrelu(0.2) for discriminator and encoder but not for generator** 
- **Be very careful when calling reshape(batch_size, -1):** The result can be very different when inputs are (batch_size, 3, 28, 28) or (batch_size, 28, 28, 3)
- **Usually bigger kernel size will be better**
- **The complexity of G and D must be relatively the same**
- **For the reconstruction L1 loss can be better** L2 loss usually cause blur
- **Adversarial loss can lead to high-quality loss**

