from transformers import  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "../models/gpt_2_base_run1/"

CONFIG_CLASS = GPT2Config
TOKENIZER_CLASS = GPT2Tokenizer
MODEL_CLASS = GPT2LMHeadModel

SPLIT_REGEX = r"[,;.!?\n](?!$)"
