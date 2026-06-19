import torch

from inference.translator import translate
from config import *
from models.transformer import TransformerTranslator
from data_utils.tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. LOAD TOKENIZER
tokenizer = Tokenizer()
tokenizer.load("data/pl_en.model")  # dostosuj jeśli masz inną nazwę

# 2. LOAD MODEL
model = TransformerTranslator(
    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN
).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])

model.eval()


while True:

    text = input("\nPL > ")

    print(
        "EN >",
        translate(
            model,
            tokenizer,
            text,
            device
        )
    )