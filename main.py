from const import VOCAB_SIZE
from models import rnn
from trainer import TrainerConfig, train


for d in range(1, 6):
    trainer_cfg = TrainerConfig(
        embedding_dim=64,
        batch_size=256,
        latent_dim=64,
        ckpt_dir="/home/trung/transformer/ckpt/rnn",
        warmstart=False,
        model=rnn.RNN(vocab_size=VOCAB_SIZE, embedding_dim=64, latent_dim=64, depth=d, cell=rnn.SimpleRNNCell),
        num_steps=1,
        context_length=128,
        learning_rate=0.01,
        warmup_steps=100,
        decay_steps=1000,
        track=False
    )
    train(trainer_cfg)
