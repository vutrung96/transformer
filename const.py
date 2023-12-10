from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
VOCAB_SIZE = tokenizer.vocab_size
