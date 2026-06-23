import math
from typing import Any, Literal

import torch
from torch import nn


# implementation of group equivariant convolutional layers
# based on Group Equivariant Convolutional Networks (Cohen & Welling, 2016) (https://arxiv.org/abs/1602.07576)
class GConv2d(nn.Module):
    S_SIZE = {"p1": 1, "p4": 4, "p4m": 8}

    def __init__(self, in_channels: int,
             out_channels: int,
             kernel_size: int | tuple[int, int],
             input_group: Literal["p1", "p4", "p4m"],
             output_group: Literal["p1", "p4", "p4m"],
             stride: int | tuple[int, int] = 1,
             padding: str | int | tuple[int, int] = 0,
             dilation: int | tuple[int, int] = 1,
             bias: bool = True,
             device: Any = None,
             dtype: Any = None):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.input_group = input_group
        self.output_group = output_group
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        factory_kwargs = {"device": device, "dtype": dtype}

        weight_shape = (self.out_channels, self.in_channels, self.S_SIZE[self.input_group], *self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        # precompute indices for the fast G-convolution trick outlined in the paper

        # u and v represent translation coordinates
        # s represents rotations for p4 group, or rotation-flips for p4m group
        # bit 0 and bit 1 are for the number of 90 deg rotations, and bit 2 is whether a flip was applied
        # this parameterisation covers cases for all groups
        # could use torch.rot90 and torch.flip instead, which would likely be faster
        # but this is easier to generalise
        def _g(s, u, v):
            s = s.float()
            u = u.float()
            v = v.float()
            r = s % 4
            m = s // 4
            elems = [[(-1) ** m * torch.cos(r * torch.pi / 2), -(-1) ** m * torch.sin(r * torch.pi / 2), u],
                     [torch.sin(r * torch.pi / 2), torch.cos(r * torch.pi / 2), v],
                     [torch.zeros_like(s), torch.zeros_like(s), torch.ones_like(s)]]

            return torch.stack([torch.stack(row, dim=-1) for row in elems], dim=-2)

        def _g_inv(g):
            u = (g[..., 0, 2]).round().long()
            v = (g[..., 1, 2]).round().long()
            r = (torch.atan2(g[..., 1, 0], g[..., 1, 1]) * 2 / torch.pi).round().long() % 4
            m = ((1 - torch.det(g)) / 2).round().long()
            return 4 * m + r, u, v

        S_in = self.S_SIZE[self.input_group]
        S_out = self.S_SIZE[self.output_group]

        s_prime, s, u, v = torch.meshgrid([torch.arange(S_out, device=device),
                                           torch.arange(S_in, device=device),
                                           torch.arange(self.kernel_size[0], device=device),
                                           torch.arange(self.kernel_size[1], device=device)],
                                          indexing="ij")

        s_bar, u_bar, v_bar = _g_inv(
            torch.inverse(_g(s_prime, torch.zeros_like(s_prime), torch.zeros_like(s_prime))) @ _g(s, u - self.kernel_size[0] // 2, v - self.kernel_size[1] // 2))
        s_bar %= S_in
        u_bar += self.kernel_size[0] // 2
        v_bar += self.kernel_size[1] // 2

        self.register_buffer("s_bar", s_bar)
        self.register_buffer("u_bar", u_bar)
        self.register_buffer("v_bar", v_bar)

    def reset_parameters(self) -> None:
        # based on the weight init for standard conv layers in Pytorch
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        S_in = self.S_SIZE[self.input_group]
        S_out = self.S_SIZE[self.output_group]

        F_plus = self.weight[:, :, self.s_bar, self.u_bar, self.v_bar].transpose(1, 2)

        input = torch.reshape(input, (*input.shape[:-4], self.in_channels * S_in, *input.shape[-2:]))
        filter = torch.reshape(F_plus, (self.out_channels * S_out, self.in_channels * S_in, *self.kernel_size))

        output = nn.functional.conv2d(input, filter, None, self.stride, self.padding, self.dilation, 1)
        output = torch.reshape(output, (*output.shape[:-3], self.out_channels, S_out, *output.shape[-2:]))

        # cannot add the bias in the conv2d call
        if self.bias is not None:
            output += self.bias[:, None, None, None]

        return output
