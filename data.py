import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
VOCAB_SIZE = tokenizer.vocab_size


@tf.py_function(Tout=tf.int32)
def tokenize_py_func(text):
    # Convert tensor to numpy array
    text = text.numpy().decode()
    # Tokenize using Hugging Face tokenizer
    return tokenizer.encode(text, return_tensors="tf")


def load_data(batch_size, seq_length):

    def split_into_sequences(x):
        # Generate a random number for dropping tokens
        d = tf.random.uniform(shape=[], minval=0, maxval=seq_length, dtype=tf.int32)
        # Drop the first d tokens and the last few tokens to make it a multiple of seq_length
        x_shape = tf.shape(x)
        begin, end = d, d + (((x_shape[1] - d) // seq_length) * seq_length)
        x = x[0, begin : end]
        # Split into sequences of specified length
        x = tf.reshape(x, (-1, seq_length))
        y = tf.roll(x, shift=-1, axis=-1)
        return {"x": x, "y": y}

    def get_data(train):
        split = "train[:95%]" if train else "train[95%:]"
        d = (
            tfds.load("wikipedia", split=split)
            .shard(num_shards=10, index=0)
            .map(lambda x: x["text"])
            .map(tokenize_py_func, num_parallel_calls=tf.data.AUTOTUNE)
            .map(split_into_sequences, num_parallel_calls=tf.data.AUTOTUNE)
            .unbatch()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return d

    return get_data(True).as_numpy_iterator(), get_data(False).as_numpy_iterator()
