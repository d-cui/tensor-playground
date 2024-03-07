#!/usr/bin/env python3

"""
Tensors are the central data abstraction in PyTorch. This study explores basic tensor initialization.
"""
from utils import pretty_print

import torch


def first_tensor():
    x = torch.empty(3, 4)

    with pretty_print("***This is our first tensor***"):
        print(type(x))  # class torch.Tensor
        print(x)  # This is a 3x4 2D matrix of zeroes
        print(x.shape)  # (3, 4)

    return x


first_tensor()


def vector():
    x = torch.empty(4)  # A 1D tensor is a vector

    with pretty_print("***This is a vector****"):
        print(x)
        print(x.shape)


vector()


def matrix():
    x = torch.empty(4, 4)  # A 2D tensor is a matrix

    with pretty_print("***This is a matrix****"):
        print(x)
        print(x.shape)


vector()


def zeros():
    x = torch.zeros(2, 3)  # Commonly want to create a tensor of all zeros

    with pretty_print("***This is a tensor of all zeros***"):
        print(x)
        print(x.shape)


zeros()


def ones():
    x = torch.ones(2, 3)  # Commonly want to create a tensor of all ones

    with pretty_print("***This is a tensor of all ones***"):
        print(x)
        print(x.shape)


ones()


def random():
    torch.manual_seed(1729)  # Seed ensures reproducibility
    x = torch.rand(2, 3)  # Commonly want to create a random initial tensor

    with pretty_print("***This is a random tensor***"):
        print(x)

        # Setting the seed again resets RNG
        torch.manual_seed(1729)
        assert torch.allclose(x, torch.rand(2, 3))


random()


def tensor_like():
    x = first_tensor()

    # Will commonly want one or more tensors to be the same shape to perform operations
    empty_like_x = torch.empty_like(x)
    zeros_like_x = torch.zeros_like(x)
    ones_like_x = torch.ones_like(x)
    rand_like_x = torch.rand_like(x)

    with pretty_print("**Create tensors like another***"):
        print(f"Original shape: {x.shape}")
        print(f"Empty shape: {empty_like_x.shape}")
        print(f"Zeroes shape: {zeros_like_x.shape}")
        print(f"Ones shape: {ones_like_x.shape}")
        print(f"Rand shape: {rand_like_x.shape}")


tensor_like()


def tensor_from_data():
    # Create a tensor from previously existing data
    # Note: torch.tensor() creates a copy of the data
    tensor_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor_tuple = torch.tensor((1, 2, 3, 4, 5, 6))
    tensor_mixed = torch.tensor(((1, 2, 3), [4, 5, 6]))

    with pretty_print("***Initialize tensors from data***"):
        print(f"Tensor from list: {tensor_list}")
        print(f"Tensor from tuple: {tensor_tuple}")
        print(f"Tensor from mixed data: {tensor_mixed}")


tensor_from_data()


def tensor_data_types():
    """
    The data type of a tensor can be set to one of the following
      * torch.bool
      * torch.int8
      * torch.uint8
      * torch.int32
      * torch.int64
      * torch.float [default]
      * torch.double
      * torch.bfloat
    """
    x = torch.ones(
        (2, 3), dtype=torch.int16
    )  # When you change the dtype, printing the tensor also specifies the dtype
    y = torch.rand(
        (2, 3), dtype=torch.double
    )  # Double is 64bit vs float default is 32bit
    z = (y * 100).to(
        torch.int32
    )  # Cast to a different dtype, *100 to have nonzero ints

    with pretty_print("***Tensor data types***"):
        print(f"int16 tensor: {x}")
        print(f"double tensor: {y}")
        print(f"Casted int32 tensor: {z}")


tensor_data_types()
