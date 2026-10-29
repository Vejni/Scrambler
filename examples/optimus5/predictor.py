"""PyTorch port of the Optimus 5-Prime ribosome-load CNN (Sample et al. 2019),
loading the Keras weights `optimusRetrainedMain.hdf5` shipped with the original repo."""
import torch
import torch.nn as nn

from scrambler_torch.convert import load_keras_weights, read_keras_h5


class Optimus5(nn.Module):
    """3x Conv1D(120, 8, same, relu) -> Flatten -> Dense(40, relu) -> Dropout(0.2) -> Dense(1)."""

    def __init__(self, seq_len=50, n_filters=120, kernel_size=8, n_hidden=40):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(c, n_filters, kernel_size, padding="same") for c in [4, n_filters, n_filters])
        self.dense = nn.Linear(seq_len * n_filters, n_hidden)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(n_hidden, 1)

    def forward(self, x):  # (N, L, 4) -> (N, 1)
        h = x.transpose(1, 2)
        for conv in self.convs:
            h = torch.relu(conv(h))
        h = h.transpose(1, 2).flatten(1)  # Keras Flatten order: (position, filter)
        return self.out(self.drop(torch.relu(self.dense(h))))

    @classmethod
    def from_keras(cls, h5_path):
        w = read_keras_h5(h5_path)  # layers in model order
        model = cls()
        convs = [k for k in w if k.startswith("conv1d")]
        denses = [k for k in w if k.startswith("dense")]
        for module, name in zip([*model.convs, model.dense, model.out], convs + denses):
            load_keras_weights(module, w[name])
        return model.eval()
