from clu import metrics
from flax.training import train_state
from flax import struct
import random
from flax import struct
from clu import metrics
from flax.training import train_state
import optax

import jax
import jax.numpy as jnp
from jax import grad, random
import optax
import orbax.checkpoint
import wandb

from models import unigram
from data import load_data
from const import VOCAB_SIZE
from flax.training import orbax_utils
from models.checkpoint_metadata import CheckpointMetadata
import orbax.checkpoint as ocp

EMBEDDING_DIM = 64
BATCH_SIZE = 64
CKPT_DIR = "/home/trung/transformer/ckpt/unigram"
WARMSTART = True
MODEL = unigram.UnigramModel(vocab_size=VOCAB_SIZE, embedding_dim=64)
NUM_STEPS = 10
CONTEXT_LENGTH = 128

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    dummy = random.randint(key1, minval=0, maxval=10, shape=(10,))
    params = module.init(rng, dummy)[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["x"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["y"]
        ).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({"params": state.params}, batch["x"])
    logits = logits.reshape(-1, logits.shape[-1])
    labels = batch["y"].reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["y"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch["y"], loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def pred_step(state, x, key, T=10):
    logits = state.apply_fn({"params": state.params}, x)
    return jax.random.categorical(key, logits / T, axis=1)


key1, key2 = random.split(random.key(0))

learning_rate = 0.01
momentum = 0.9

init_rng = jax.random.key(0)
state = create_train_state(MODEL, init_rng, learning_rate, momentum)
ckpt_metadata = CheckpointMetadata(wandb_id="", step=0, cfg=MODEL.cfg())
metrics_history = {"train_loss": [], "train_accuracy": []}

options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3, create=True)
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    CKPT_DIR, orbax_checkpointer, options
)
step = 0

if WARMSTART:
    print("Restoring from checkpoint")
    latest_step = checkpoint_manager.latest_step()
    print("Latest step ", latest_step)
    target = {"model": state, "metadata": ckpt_metadata.to_dict()}
    # print("Abstract ckpt: ", abstract_ckpt)
    restored_ckpt = checkpoint_manager.restore(
        checkpoint_manager.latest_step(),
        items = target
    )
    step = latest_step
    state = restored_ckpt['model']
    assert restored_ckpt['metadata']['cfg'] == MODEL.cfg()
    ckpt_metadata = CheckpointMetadata.from_dict(restored_ckpt['metadata'])
    print("Restored model: ", state)
else:
    print("Starting from scratch")
    wandb_id = wandb.util.generate_id()
    ckpt_metadata.wandb_id = wandb_id

wandb.init(
    # set the wandb project where this run will be logged
    project="transformer",
    resume="allow",
    id=ckpt_metadata.wandb_id,
    # track hyperparameters and run metadata
    config=MODEL.cfg(),
)

d_iter = load_data(batch_size=BATCH_SIZE, seq_length=CONTEXT_LENGTH)

# Training loop
for i in range(NUM_STEPS):

    step += 1
    print("Step: ", step)
    
    print("Loading batch ")
    # Get the next batch from the training dataset
    train_batch = next(d_iter)

    print("Running train step")
    # Run optimization steps over training batches and compute batch metrics
    state = train_step(
        state, train_batch
    )  # get updated train state (which contains the updated parameters)
    state = compute_metrics(state=state, batch=train_batch)  # aggregate batch metrics

    print("Done running train step")
    if step % 1000 == 0:
        metrics_dict = {}
        for metric, value in state.metrics.compute().items():  # compute metrics
            metrics_dict[f"{metric}"] = value  # record metrics
        wandb.log(metrics_dict)

    if step % 10000 == 0:
        ckpt = {"model": state, "metadata": ckpt_metadata.to_dict()}
        save_args = orbax_utils.save_args_from_target(ckpt)
        checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})
    ckpt_metadata.step = step


wandb.finish()
