import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset

from config import *

from metrics.bleu import evaluate_bleu
from data_utils.tokenizer import Tokenizer

from metrics.validation import validate
from metrics.perplexity import perplexity

from inference.translator import translate

from data_utils.dataset import TranslationDataset
from data_utils.collate import collate_fn

from models.transformer import TransformerTranslator
from models.masks import create_causal_mask

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Device:", device)

dataset = load_dataset("opus100", "en-pl")

#train_data = TranslationDataset(dataset["train"])
train_split = dataset["train"].select(range(200000))  # zamiast 1M

train_data = TranslationDataset(train_split)

loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

valid_split = dataset["validation"]

valid_data = TranslationDataset(valid_split)

valid_loader = DataLoader(
    valid_data,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

model = TransformerTranslator(

    src_vocab=VOCAB_SIZE,
    tgt_vocab=VOCAB_SIZE,

    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    max_len=MAX_LEN

).to(device)

tokenizer = Tokenizer()
tokenizer.load(TOKENIZER_PATH)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


import os

start_epoch = 0

if os.path.exists(CHECKPOINT):

    print("Ładowanie checkpointu...")

    checkpoint = torch.load(CHECKPOINT, map_location=device)

    model.load_state_dict(checkpoint["model"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    start_epoch = checkpoint["epoch"] + 1



for epoch in range(start_epoch, EPOCHS):

    model.train()

    loop = tqdm(loader, desc=f"Epoch {epoch}")

    total_loss = 0

    for step, (src, tgt) in enumerate(loop):

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        tgt_mask = create_causal_mask(
            tgt_input.size(1)
        ).to(device)

        logits = model(
            src,
            tgt_input,
            src_mask=None,
            tgt_mask=tgt_mask
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 200 == 0:
            print(f"\n[Epoch {epoch} | Step {step}] loss = {loss.item():.4f}")

        loop.set_postfix(loss=f"{loss.item():.4f}")

        #loop.set_postfix(loss=loss.item())
        loop.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_train_loss = total_loss / len(loader)

    val_loss, val_acc = validate(
        model,
        valid_loader,
        criterion,
        device
    )

    bleu = evaluate_bleu(
        model,
        tokenizer,
        dataset["validation"],
        device,
        num_examples=100
    )

    print("=" * 40)

    print(f"Epoch {epoch}")

    print(f"Train loss      : {avg_train_loss:.4f}")

    print(f"Validation loss : {val_loss:.4f}")

    print(f"Accuracy        : {val_acc * 100:.2f}%")

    print(f"Perplexity      : {perplexity(val_loss):.2f}")

    print(f"BLEU            : {bleu:.2f}")

    sample = dataset["validation"][0]

    pl = sample["translation"]["pl"]

    gt = sample["translation"]["en"]

    pred = translate(
        model,
        tokenizer,
        pl,
        device
    )

    print()

    print("Example")

    print("PL    :", pl)

    print("GT    :", gt)

    print("MODEL :", pred)

    print()

    print("=" * 40)


    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        },
        CHECKPOINT
    )


print("Trening zakończony")
