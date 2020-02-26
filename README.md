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