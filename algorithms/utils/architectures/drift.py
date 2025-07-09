import flax.linen as nn


class MetaDrift(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size, use_bias=False)(x)
        x = nn.tanh(x)
        x = nn.Dense(1, use_bias=False)(x)

        return x
