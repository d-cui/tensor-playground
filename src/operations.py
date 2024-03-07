#!/usr/bin/env python3

"""
Tensors are the central data abstraction in PyTorch. This study explores mathematical operations on tensors.
"""

from utils import pretty_print

import torch


def tensor_scalar_arithmetic():
    """
    You can apply basic arithmetic between tensors and scalars. Arithmetic is simply distributed element-wise across the tensor.
    """
    ones = torch.zeros(2, 2) + 1
    twos = torch.ones(2, 2) * 2
    threes = (torch.ones(2, 2) * 7 - 1) / 2
    fours = twos**2
    sqrt2s = twos**0.5

    with pretty_print("***Tensor<>scalar arithmetic***"):
        print(f"Zeroes + 1: {ones}")
        print(f"Ones * 2: {twos}")
        print(f"Multiply/subtract/divide: {threes}")
        print(f"Squared: {fours}")
        print(f"Sqrt: {sqrt2s}")


tensor_scalar_arithmetic()


def tensor_tensor_arithmetic():
    """
    You can apply basic arithmetic to two scalars. When they have the same shape, the outputs are distributed element-wise.
    """
    twos = torch.ones(2, 2) * 2
    powers2 = twos ** torch.tensor([[1, 2], [3, 4]])

    ones = torch.ones(2, 2)
    fours = twos**2
    fives = ones + fours

    eights = fours * twos

    with pretty_print("***Tensor<>tensor arithmetic***"):
        print(f"Powers of 2: {powers2}")
        print(f"Fives: {fives}")
        print(f"Eithers: {eights}")


tensor_tensor_arithmetic()


def tensor_tensor_diff_shape():
    """
    You can't generally apply arithmetic to two tensors with different shapes.
    """
    x = torch.rand(2, 3)
    y = torch.rand(3, 2)

    try:
        z = x * y
    except RuntimeError as e:
        with pretty_print("***Tensor arithmetic different shapes***"):
            print(f"Can't multiply x and y due to shape mismatch.")


tensor_tensor_diff_shape()
