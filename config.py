# ==========================================
# Tokenizer
# ==========================================

VOCAB_SIZE = 16000

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


# ==========================================
# Transformer
# ==========================================

D_MODEL = 256

NUM_HEADS = 8

NUM_LAYERS = 4

D_FF = 1024

MAX_LEN = 512


# ==========================================
# Trening
# ==========================================

BATCH_SIZE = 64

LEARNING_RATE = 1e-4

EPOCHS = 6


# ==========================================
# Dane
# ==========================================

DATASET_NAME = "opus_books"

LANGUAGE_PAIR = "pl-en"


# ==========================================
# Ścieżki
# ==========================================

TOKENIZER_MODEL = "data/pl_en.model"

TOKENIZER_VOCAB = "data/pl_en.vocab"

TOKENIZER_TEXT = "data/train_text.txt"

TOKENIZER_PATH = "data/pl_en.model"

CHECKPOINT = "checkpoints/translator.pt"