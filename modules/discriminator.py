
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator for adversarial training and multi-task learning
    """
    def __init__(self, feat_dim, num_class):
        super(Discriminator, self).__init__()

        self.linear = nn.Linear(feat_dim, num_class)

        self.copy_grad = None
    
    def forward(self, inputs):
        """
        args:
            inputs: B x H
        output:
            predictions: B x C
        """
        predictions = self.linear(inputs)
        return predictions

    # Init copy_grad
    def init_copy_grad_(self):
        self.copy_grad = []
        for param in self.parameters():
            self.copy_grad.append(torch.zeros(param.shape, device=param.device, requires_grad=False))

    # Zeros copy_grad
    def zero_copy_grad(self):
        if self.copy_grad is None:
            self.init_copy_grad_()
        else:
            for i in range(len(self.copy_grad)):
                self.copy_grad[i] -= self.copy_grad[i]

    # Add model grad to copy_grad
    def add_copy_grad(self):
        if self.copy_grad is None:
            self.copy_grad = self.init_copy_grad_()

        for i, param in enumerate(self.parameters()):
            self.copy_grad[i].data += param.grad.data

    # Copy model grad to copy_grad
    def to_copy_grad(self):
        if self.copy_grad is None:
            self.copy_grad = self.init_copy_grad_()

        for i, param in enumerate(self.parameters()):
            self.copy_grad[i].data.copy_(param.grad)
            
    # Copy copy_grad to model grad
    def from_copy_grad(self):
        if self.copy_grad is None:
            self.copy_grad = self.init_copy_grad_()

        for i, param in enumerate(self.parameters()):
            param.grad.data.copy_(self.copy_grad[i])
