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


def load_data(batch_size):

  d = (tfds.load('wikipedia')['train']
     .map(lambda x: x['text'], num_parallel_calls=tf.data.AUTOTUNE)
     .map(tokenize_py_func, num_parallel_calls=tf.data.AUTOTUNE)
     .map(lambda x: {'x': x[0][:-1], 'y': tf.roll(x[0], -1, 0)[:-1]}, num_parallel_calls=tf.data.AUTOTUNE)
     .unbatch() 
     .shuffle(1024)
     .batch(batch_size)
     .prefetch(1)
    )

  return d.as_numpy_iterator()
