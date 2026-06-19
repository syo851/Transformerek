import torch

from config import *


def translate(
    model,
    tokenizer,
    text,
    device,
    max_len=MAX_LEN
):

    model.eval()

    src = tokenizer.encode(text)

    src = torch.tensor(
        src,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    tgt = [BOS_ID]

    with torch.no_grad():

        for _ in range(max_len):

            tgt_tensor = torch.tensor(
                tgt,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            logits = model(
                src,
                tgt_tensor
            )

            next_token = logits[
                0,
                -1
            ].argmax().item()

            tgt.append(next_token)

            if next_token == EOS_ID:
                break

    return tokenizer.decode(tgt)