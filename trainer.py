import jax
import jax.numpy as jnp
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
    b1: float
    debug_grad: bool


@struct.dataclass
class Metrics(metrics.Collection):
    train_loss: metrics.Average.from_output("train_loss")
    grad_norm_d0_i0: metrics.Average.from_output("grad_norm_d0_i0")
    grad_norm_d0_i30: metrics.Average.from_output("grad_norm_d0_i30")
    grad_norm_d1_i0: metrics.Average.from_output("grad_norm_d1_i0")
    grad_norm_d1_i30: metrics.Average.from_output("grad_norm_d1_i30")


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
    rng, subkey = jax.random.split(rng)
    variables = trainer_cfg.model.init(subkey, dummy)
    params, perturbations = variables["params"], variables["perturbations"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=trainer_cfg.learning_rate,
        warmup_steps=trainer_cfg.warmup_steps,
        decay_steps=trainer_cfg.decay_steps,
        end_value=0.0,
    )
    tx = optax.adamw(learning_rate=schedule, b1=trainer_cfg.b1)
    return (
        TrainState.create(
            apply_fn=trainer_cfg.model.apply,
            params=params,
            tx=tx,
            metrics=Metrics.empty(),
        ),
        perturbations,
    )


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["x"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["y"]
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, logits


@jax.jit
def train_step_debug(state, perturbations, batch):
    """Train for a single step."""

    def loss_fn(params, perturbations):
        logits = state.apply_fn(
            {"params": params, "perturbations": perturbations}, batch["x"]
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["y"]
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)
    (_, logits), (grads, grads_dbg) = grad_fn(state.params, perturbations)
    state = state.apply_gradients(grads=grads)
    return state, logits, grads_dbg
    

@jax.jit
def compute_metrics(logits, state, batch, grads_dbg=None):
    labels = batch["y"]
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    grad_norm_d0_i0 = jax.numpy.linalg.norm(grads_dbg["state_cell_0_0"], axis=1)
    grad_norm_d0_i30 = jax.numpy.linalg.norm(grads_dbg["state_cell_0_30"], axis=1)
    grad_norm_d1_i0 = jax.numpy.linalg.norm(grads_dbg["state_cell_1_0"], axis=1)
    grad_norm_d1_i30 = jax.numpy.linalg.norm(grads_dbg["state_cell_1_30"], axis=1)
    metric_updates = state.metrics.single_from_model_output(
        grad_norm_d0_i0=grad_norm_d0_i0,
        grad_norm_d0_i30=grad_norm_d0_i30,
        grad_norm_d1_i0=grad_norm_d1_i0,
        grad_norm_d1_i30=grad_norm_d1_i30,
        train_loss=loss,
        eval_loss = jnp.nan
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({"params": state.params}, batch["x"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["y"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        grad_norm_d0_i0=jnp.nan,
        grad_norm_d0_i30=jnp.nan,
        grad_norm_d1_i0=jnp.nan,
        grad_norm_d1_i30=jnp.nan,
        train_loss = jnp.nan,
        eval_loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state


@jax.jit
def pred_step(state, x, key, T=10):
    logits = state.apply_fn({"params": state.params}, x)
    return jax.random.categorical(key, logits / T, axis=1)


def train(trainer_cfg):

    init_rng = jax.random.key(0)

    state, perturbations = create_train_state(init_rng, trainer_cfg)
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
        restored_ckpt = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), items=target
        )
        step = latest_step
        state = restored_ckpt["model"]
        assert restored_ckpt["metadata"]["cfg"] == trainer_cfg.model.cfg()
        ckpt_metadata = CheckpointMetadata.from_dict(restored_ckpt["metadata"])
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

    d_iter, d_test_iter = load_data(
        batch_size=trainer_cfg.batch_size, seq_length=trainer_cfg.context_length
    )

    print("Starting training")

    # Training loop
    for i in range(trainer_cfg.num_steps):

        step += 1

        # Get the next batch from the training dataset
        train_batch = next(d_iter)

        # Run optimization steps over training batches and compute batch metrics
        if trainer_cfg.debug_grad:
            state, logits, grads_dbg  = train_step_debug(state, perturbations, train_batch)
        else:
            state, logits = train_step(
                state, train_batch
            )  # get updated train state (which contains the updated parameters)
        state = compute_metrics(
            logits,
            state,
            train_batch,
            grads_dbg=grads_dbg if trainer_cfg.debug_grad else None,
        )  # aggregate batch metrics
        if step % 100 == 0:
            print("Step: ", step)
            state = eval_step(state, next(d_test_iter))  # evaluate on test set
            metrics_dict = {}
            for metric, value in state.metrics.compute().items():  # compute metrics
                metrics_dict[f"{metric}"] = value  # record metrics
            if trainer_cfg.track:
                wandb.log(data=metrics_dict, step=step)
            state.replace(metrics=Metrics.empty())

        if step % 10000 == 0:
            ckpt = {"model": state, "metadata": ckpt_metadata.to_dict()}
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(step, ckpt, save_kwargs={"save_args": save_args})
        ckpt_metadata.step = step

    if trainer_cfg.track:
        wandb.finish()
