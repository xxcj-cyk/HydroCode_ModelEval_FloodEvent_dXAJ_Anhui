import torch.nn as nn
from torch.nn import functional as F


def parse_hidden_sizes(hidden_size):
    if isinstance(hidden_size, int):
        sizes = [hidden_size]
    else:
        sizes = [int(h) for h in hidden_size]
    if not sizes or any(h < 1 for h in sizes):
        raise ValueError('hidden_size must be a positive int or list of positive ints')
    return sizes


def parse_dropout_rates(dropout, n_layers):
    if isinstance(dropout, (list, tuple)):
        rates = [float(p) for p in dropout]
        if len(rates) != n_layers:
            raise ValueError(f'len(dropout) ({len(rates)}) != n_layers ({n_layers})')
    else:
        rates = [float(dropout)] * n_layers
    if any(not 0.0 <= p < 1.0 for p in rates):
        raise ValueError(f'dropout must be in [0, 1), got {rates}')
    return rates


def normalize_input_proj(input_proj):
    if input_proj is None or input_proj == 'none':
        return None
    if input_proj == 'linear':
        return 'linear'
    if input_proj in ('linear_relu', 'relu'):
        return 'linear_relu'
    raise ValueError(
        f"input_proj must be None, 'none', 'linear', or 'linear_relu', got {input_proj!r}"
    )


def lstm_in_features(layer_index, input_size, hidden_sizes, has_input_proj):
    if layer_index == 0:
        return hidden_sizes[0] if has_input_proj else input_size
    return hidden_sizes[layer_index - 1]


class SeqRegLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        dropout=0.0,
        input_proj=None,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        hidden_sizes = parse_hidden_sizes(hidden_size)
        self.hidden_sizes = hidden_sizes

        proj_mode = normalize_input_proj(input_proj)
        self.input_proj_mode = proj_mode
        self.input_linear = (
            nn.Linear(self.input_size, hidden_sizes[0]) if proj_mode else None
        )

        self.lstm_layers = nn.ModuleList(
            nn.LSTM(
                input_size=lstm_in_features(i, self.input_size, hidden_sizes, proj_mode),
                hidden_size=h,
                batch_first=True,
            )
            for i, h in enumerate(hidden_sizes)
        )
        self.dropout_rates = parse_dropout_rates(dropout, len(hidden_sizes))
        self.output_linear = nn.Linear(hidden_sizes[-1], self.output_size)

    def forward(self, x):
        if self.input_linear is not None:
            x = self.input_linear(x)
            if self.input_proj_mode == 'linear_relu':
                x = F.relu(x)
        for lstm, rate in zip(self.lstm_layers, self.dropout_rates):
            x, _ = lstm(x)
            if rate > 0:
                x = F.dropout(x, p=rate, training=self.training)
        return self.output_linear(x)
