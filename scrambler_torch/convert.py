"""Load weights of Keras (2.2.x) scramblers and simple predictors into PyTorch.

Only h5py is needed; TensorFlow/Keras are not required.
"""
import h5py
import numpy as np
import torch


def read_keras_h5(path):
    """{layer_name: [weight arrays]} from a Keras .h5 (full model or weights-only)."""
    out = {}
    with h5py.File(path, "r") as f:
        g = f["model_weights"] if "model_weights" in f else f
        for name in g.attrs["layer_names"]:
            name = name.decode() if isinstance(name, bytes) else name
            weight_names = [w.decode() if isinstance(w, bytes) else w for w in g[name].attrs["weight_names"]]
            if weight_names:
                out[name] = [np.array(g[name][w]) for w in weight_names]
    return out


def conv_kernel(kernel):
    """Keras Conv1D (k, in, out) or 1-row Conv2D (1, k, in, out) kernel -> torch (out, in, k)."""
    kernel = kernel.reshape(kernel.shape[-3:]) if kernel.ndim == 4 else kernel
    return torch.as_tensor(np.ascontiguousarray(kernel.transpose(2, 1, 0)))


def dense_kernel(kernel):
    """Keras Dense (in, out) -> torch Linear (out, in)."""
    return torch.as_tensor(np.ascontiguousarray(kernel.T))


def _set(param, value):
    with torch.no_grad():
        param.copy_(torch.as_tensor(value).reshape(param.shape))


def keras_tensors(module):
    """Torch tensors of a module in Keras weight order."""
    if isinstance(module, torch.nn.InstanceNorm1d):  # InstanceNormalization: [beta, gamma]
        return [module.bias, module.weight]
    if isinstance(module, torch.nn.BatchNorm1d):  # [gamma, beta, moving_mean, moving_variance]
        return [module.weight, module.bias, module.running_mean, module.running_var]
    return [module.weight] + ([module.bias] if module.bias is not None else [])  # conv: [kernel, bias]


def scrambler_layer_map(net, prefix=""):
    """{Keras layer name: torch module} for a ScramblerNet."""
    p = prefix + "scrambler_"
    layers = {p + "conv_0": net.conv0, p + "last_block_conv": net.last_block_conv, p + "final_conv": net.final_conv}
    for g, (skip_conv, group) in enumerate(zip(net.skip_convs, net.groups)):
        layers[f"{p}skip_conv_{g}"] = skip_conv
        for l, block in enumerate(group):
            stem = f"{p}resblock_{g}_{l}_"
            layers.update({stem + "conv_0": block.conv0, stem + "conv_1": block.conv1})
            if not isinstance(block.norm0, torch.nn.Identity):
                layers.update({stem + "norm_0": block.norm0, stem + "norm_1": block.norm1})
    return layers


def load_keras_weights(module, weights):
    """Copy one Keras layer's weight list into a conv/norm/linear module."""
    for t, w in zip(keras_tensors(module), weights):
        if isinstance(module, torch.nn.Conv1d) and t is module.weight:
            w = conv_kernel(w)
        elif isinstance(module, torch.nn.Linear) and t is module.weight:
            w = dense_kernel(w)
        _set(t, w)


def load_keras_scrambler_weights(net, weights, prefix=""):
    """Copy Keras scrambler weights ({layer: [arrays]}, e.g. from `read_keras_h5`) into a ScramblerNet."""
    for name, module in scrambler_layer_map(net, prefix).items():
        load_keras_weights(module, weights[name])
    return net
