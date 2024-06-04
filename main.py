from const import VOCAB_SIZE
from models import rnn
from trainer import TrainerConfig, train


trainer_cfg = TrainerConfig(
    embedding_dim=64,
    batch_size=256,
    latent_dim=64,
    ckpt_dir="/home/trung/transformer/ckpt/rnn",
    warmstart=False,
    model=rnn.RNN(vocab_size=VOCAB_SIZE, embedding_dim=64, latent_dim=64, depth=2, cell=rnn.SimpleRNNCell),
    num_steps=100000,
    context_length=32,
    learning_rate=0.02,
    warmup_steps=100,
    decay_steps=1000,
    b1=0.9,
    track=True,
    debug_grad=True
)
train(trainer_cfg)
