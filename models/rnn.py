from flax import linen as nn
import jax
import jax.numpy as jnp
from models.checkpoint_metadata import ModelType
from typing import Type


class BaseRNNCell(nn.Module):
    latent_dim: int


class SimpleRNNCell(BaseRNNCell):
    latent_dim: int

    @nn.compact
    def __call__(self, x, state):
        # Shape of x: (batch_size, embedding_dim)
        # Shape of state: (batch_size, latent_dim)
        state = nn.Dense(
            features=self.latent_dim,
            dtype=jnp.bfloat16,
        )(state) + nn.Dense(
            features=self.latent_dim,
            dtype=jnp.bfloat16,
        )(
            x
        )
        state = nn.activation.tanh(state)
        x = nn.Dense(
            features=x.shape[1], dtype=jnp.bfloat16
        )(state)
        return x, state


class RNN(nn.Module):
    vocab_size: int
    embedding_dim: int
    latent_dim: int
    depth: int
    cell: Type[BaseRNNCell]

    def init_state(self, batch_size):
        return jnp.zeros((batch_size, self.latent_dim), dtype=jnp.bfloat16)

    @nn.compact
    def __call__(self, x, state=None):
        # Shape of x: (batch_size, seq_len)
        if state is None:
            state = self.init_state(x.shape[0])

        # Embed input words
        x = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            name="embed",
            dtype=jnp.bfloat16,
        )(x)

        # Run RNN with depth d
        for d in range(self.depth):
            xs = []
            cell = self.cell(latent_dim=self.latent_dim)
            for i in range(x.shape[1]):
                x_slice, state = cell(x[:, i, :], state)
                state = self.perturb(f"state_cell_{d}_{i}", state)
                x_slice = self.perturb(f"x_cell_{d}_{i}", x_slice)
                xs.append(x_slice)
            state = self.init_state(x.shape[0])
            x = jnp.stack(xs, axis=1)

        # Convert output to vocabulary logit
        y = nn.Dense(features=self.vocab_size, dtype=jnp.bfloat16, name="unembed")(x)

        return y

    def cfg(self):
        return {
            "model_type": ModelType.RNN.value,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.latent_dim,
            "depth": self.depth,
        }

