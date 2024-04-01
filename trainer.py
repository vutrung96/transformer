import jax
import optax
import wandb
from clu import metrics
from data import load_data
from flax import struct
from flax import linen as nn
from flax.training import orbax_utils
from flax.training import train_state
from models.checkpoint_metadata import CheckpointMetadata
import orbax.checkpoint as ocp


@struct.dataclass
class TrainerConfig:
    embedding_dim: int
    batch_size: int
    latent_dim: int
    ckpt_dir: str
    warmstart: bool
    model: nn.Module
    num_steps: int
    context_length: int
    learning_rate: float
    warmup_steps: int
    decay_steps: int
    track: bool


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(rng, trainer_cfg):
    """Creates an initial `TrainState`."""
    dummy = jax.random.randint(
        rng,
        minval=0,
        maxval=10,
        shape=(trainer_cfg.batch_size, trainer_cfg.context_length),
    )
    params = trainer_cfg.model.init(rng, dummy)[
        "params"
    ]  # initialize parameters by passing a template image
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=trainer_cfg.learning_rate,
        warmup_steps=trainer_cfg.warmup_steps,
        decay_steps=trainer_cfg.decay_steps,
        end_value=0.0,
    )
    tx = optax.adamw(learning_rate=schedule)
    return TrainState.create(
        apply_fn=trainer_cfg.model.apply, params=params, tx=tx, metrics=Metrics.empty()
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
        logits=logits, labels=labels
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def pred_step(state, x, key, T=10):
    logits = state.apply_fn({"params": state.params}, x)
    return jax.random.categorical(key, logits / T, axis=1)


def train(trainer_cfg):

    init_rng = jax.random.key(0)

    state = create_train_state(
        init_rng,
        trainer_cfg
    )
    ckpt_metadata = CheckpointMetadata(wandb_id="", step=0, cfg=trainer_cfg.model.cfg())

    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        trainer_cfg.ckpt_dir, orbax_checkpointer, options
    )
    step = 0

    if trainer_cfg.warmstart:
        print("Restoring from checkpoint")
        latest_step = checkpoint_manager.latest_step()
        print("Latest step ", latest_step)
        target = {"model": state, "metadata": ckpt_metadata.to_dict()}
        # print("Abstract ckpt: ", abstract_ckpt)
        restored_ckpt = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), items=target
        )
        step = latest_step
        state = restored_ckpt["model"]
        assert restored_ckpt["metadata"]["cfg"] == trainer_cfg.model.cfg()
        ckpt_metadata = CheckpointMetadata.from_dict(restored_ckpt["metadata"])
        # print("Restored model: ", state)
    else:
        print("Starting from scratch")
        wandb_id = wandb.util.generate_id()
        ckpt_metadata.wandb_id = wandb_id

    logged_cfg = trainer_cfg.model.cfg()
    logged_cfg["scheduler"] = "cosine_warmup"
    logged_cfg["learning_rate"] = trainer_cfg.learning_rate
    logged_cfg["warmup_steps"] = trainer_cfg.warmup_steps
    logged_cfg["decay_steps"] = trainer_cfg.decay_steps
    logged_cfg["batch_size"] = trainer_cfg.batch_size

    if trainer_cfg.track:
        wandb.init(
            # set the wandb project where this run will be logged
            project="transformer",
            resume="allow",
            id=ckpt_metadata.wandb_id,
            # track hyperparameters and run metadata
            config=logged_cfg,
        )

    d_iter = load_data(
        batch_size=trainer_cfg.batch_size, seq_length=trainer_cfg.context_length
    )

    print("Starting training")

    # Training loop
    for i in range(trainer_cfg.num_steps):

        step += 1

        # Get the next batch from the training dataset
        train_batch = next(d_iter)

        # Run optimization steps over training batches and compute batch metrics
        state = train_step(
            state, train_batch
        )  # get updated train state (which contains the updated parameters)
        state = compute_metrics(
            state=state, batch=train_batch
        )  # aggregate batch metrics
        if step % 100 == 0:
            print("Step: ", step)
            metrics_dict = {}
            for metric, value in state.metrics.compute().items():  # compute metrics
                metrics_dict[f"{metric}"] = value  # record metrics
            if trainer_cfg.track:
                wandb.log(data=metrics_dict, step=step)

        if step % 10000 == 0:
            ckpt = {"model": state, "metadata": ckpt_metadata.to_dict()}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})
        ckpt_metadata.step = step

    if trainer_cfg.track:
        wandb.finish()
