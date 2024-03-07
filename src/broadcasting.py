#!/usr/bin/env python3

"""
Tensors are the central data abstraction in PyTorch. This study explores tensor broadcasting, or performing operations between tensors of different shapes.

Why broadcasting?

In deep learning, it's common to multiple a tensor of weights by a batch of input tensors (independent inputs, processed in parallel for efficiency). The operation is applied to EACH input tensor SEPARATELY, returning an identically shaped tensor.

For example, if the batched input tensor has shape (A, B), or there are A input tensors of shape (1, B). Given a weight tensor of dimension (1, B), we would like (A, B) * (1, B) to output a SAME DIMENSION tensor as the inputs, or (A, B). In other words, each input instance is independently multiplied by the (1, B) weight vector.

Rules for broadcasting are
  * Each tensor must have at least one dimension
  * Right align the two tensors
    - i.e. (2, 3, 4) and (_, 3, 4)
  * Compare the dimension sizes of the two tensors, going from last to first (right to left)
    - Each dimension must be equal or
    - One of the dimensions must be size 1 or
    - The dimension does not exist in one of the tensors

See the examples below.
"""

from utils import pretty_print

import torch


def simple_broadcasting():
    """
    This is a simple example illustrating multipling a (2, 4) tensor with a (1, 4) tensor.
    """
    rand = torch.rand(2, 4)
    doubled = rand * (torch.ones(1, 4) * 2)

    assert torch.allclose(rand * 2, doubled)

    with pretty_print("***Simple tensor broadcast***"):
        print(f"Random 2x4 tensor: {rand}")
        print(f"Multiplied by 1x4 tensor of 2s: {doubled}")


simple_broadcasting()


def multi_dim_broadcasting():
    """
    This is an example exploring multi-dimensional broadcasting.
    """
    a = torch.ones(4, 3, 2)
    b = a * (
        torch.tensor([[1, 2], [3, 4], [5, 6]])
    )  # 2nd and 3rd dimensions are identical, so should be broadcast across the 4 layers

    print(b)

    c = a * torch.tensor(
        [[1], [2], [3]]
    )  # 2nd dim is 3 and 3rd dim is 1, broadcast across the layers and columns, so every layer and column are identical

    print(c)

    d = a * torch.tensor(
        [[1, 2]]
    )  # 2nd dim is 1 and 3rd dim is 2, broadcast across the layers and rows, so every layer and row are identical

    print(d)


multi_dim_broadcasting()
