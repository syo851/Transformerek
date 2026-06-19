import torch

from config import PAD_ID


def token_accuracy(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T)
    """

    predictions = logits.argmax(dim=-1)

    mask = targets != PAD_ID

    correct = (predictions == targets) & mask

    accuracy = correct.sum().float() / mask.sum().float()

    return accuracy.item()