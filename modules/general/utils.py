import torch
import torch.nn as nn


def normalization(channels, groups=32):
    r"""Make a standard normalization layer, i.e. GroupNorm.
    Args:
        channels: number of input channels.
        groups: number of groups for group normalization.
    Returns:
        a ``nn.Module`` for normalization.
    """
    return nn.GroupNorm(groups, channels)


def Linear(*args, **kwargs):
    r"""Wrapper of ``nn.Linear`` with kaiming_normal_ initialization."""
    layer = nn.Linear(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv1d(*args, **kwargs):
    r"""Wrapper of ``nn.Conv1d`` with kaiming_normal_ initialization."""
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv2d(*args, **kwargs):
    r"""Wrapper of ``nn.Conv2d`` with kaiming_normal_ initialization."""
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def zero_module(module):
    r"""Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    r"""Scale the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    r"""Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def append_dims(x, target_dims):
    r"""Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def checkpoint(func, inputs, params, flag):
    r"""Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    Args:
        func: the function to evaluate.
        inputs: the argument sequence to pass to `func`.
        params: a sequence of parameters `func` depends on but does not
                explicitly take as arguments.
        flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
