from flax import linen as nn
import jax
from models.checkpoint_metadata import ModelType


class UnigramModel(nn.Module):
    vocab_size: int
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Embed(
            num_embeddings=self.vocab_size, features=self.embedding_dim, name="unigram"
        )(x)
        x = nn.Dense(features=self.vocab_size)(x)
        return x

    def cfg(self):
        return {
            "model_type": ModelType.UNIGRAM.value,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "activation": "relu",
        }
