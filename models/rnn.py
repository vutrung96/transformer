from flax import linen as nn
import jax
import jax.numpy as jnp
from models.checkpoint_metadata import ModelType


class RNN(nn.Module):
    vocab_size: int
    embedding_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, x, state=None):
        if state is None:
            state = jnp.zeros((x.shape[0], self.latent_dim))
        # Shape of x: (batch_size, seq_len)
        x = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embedding_dim, name="embed"
        )(x)
        ys = []
        for i in range(x.shape[1]):
            state = nn.Dense(features=self.latent_dim)(state) + nn.Dense(
                features=self.latent_dim
            )(x[:, i, :])
            state = nn.relu(state)
            y = nn.Dense(features=self.vocab_size)(state)
            ys.append(y)
        ys = jnp.stack(ys, axis=1)
        return ys

    def cfg(self):
        return {
            "model_type": ModelType.RNN.value,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "latent_dim": self.latent_dim,
            "input_dense_layer": 1,
            "state_dense_layer": 1,
            "output_dense_layer": 1,
        }
