from clu import metrics
from flax.training import train_state
from flax import struct
import jax
import optax

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    dummy = random.randint(key1, minval=0, maxval=10, shape=(10,))
    params = module.init(rng, dummy)['params'] # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['x'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['y']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state 

@jax.jit 
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['x'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['y']).mean()
    metric_updates = state.metrics.single_from_model_output(
    logits=logits, labels=batch['y'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state
