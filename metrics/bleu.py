from sacrebleu import corpus_bleu

from inference.translator import translate


def evaluate_bleu(
        model,
        tokenizer,
        dataset,
        device,
        num_examples=100
):

    predictions = []

    references = []

    for sample in dataset.select(range(num_examples)):

        pl = sample["translation"]["pl"]

        gt = sample["translation"]["en"]

        pred = translate(
            model,
            tokenizer,
            pl,
            device
        )

        predictions.append(pred)

        references.append([gt])

    bleu = corpus_bleu(
        predictions,
        references
    )

    return bleu.score