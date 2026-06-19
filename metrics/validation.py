import torch

from metrics.accuracy import token_accuracy


def validate(
        model,
        loader,
        criterion,
        device):

    model.eval()

    total_loss = 0
    total_acc = 0

    with torch.no_grad():

        for src, tgt in loader:

            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            logits = model(
                src,
                tgt_input
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_target.reshape(-1)
            )

            total_loss += loss.item()

            total_acc += token_accuracy(
                logits,
                tgt_target
            )

    model.train()

    avg_loss = total_loss / len(loader)

    avg_acc = total_acc / len(loader)

    return avg_loss, avg_acc