import argparse
from pathlib import Path

import torch

from functions import eval_model_saved, load_checkpoint, run
from utils import make_dataloaders


# Expose the training, evaluation and backbone-selection options through the CLI.
def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=["gpt2", "bert", "both"], default="both")
    parser.add_argument("--dataset-dir", default="dataset/ATIS")
    parser.add_argument("--output-dir", default="outputs")

    parser.add_argument("--gpt2-model-name", default="openai-community/gpt2")
    parser.add_argument("--bert-model-name", default="google-bert/bert-base-uncased")

    parser.add_argument("--gpt2-lr", type=float, default=5e-5)
    parser.add_argument("--bert-lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--dev-size", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate a saved checkpoint without training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the saved checkpoint used with --eval.",
    )

    return parser.parse_args()


def print_model_info(model_type, model_name, train_loader, dev_loader, test_loader, lang):
    print("=" * 89)
    print("Model:", model_type)
    print("Checkpoint:", model_name)
    print("Train samples:", len(train_loader.dataset))
    print("Dev samples:", len(dev_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Slot labels:", len(lang.slot2id))
    print("Intent labels:", len(lang.intent2id))
    print("=" * 89)


# Build a separate tokenizer and collation pipeline for the selected pretrained backbone.
def make_model_dataloaders(args, model_type, model_name, device):
    return make_dataloaders(
        device=device,
        model_type=model_type,
        model_name=model_name,
        dataset_dir=args.dataset_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        data_seed=args.data_seed,
        dev_size=args.dev_size,
        max_length=args.max_length,
    )


# Train and evaluate repeated runs for one backbone, writing results under outputs/<model_type>/.
def run_model(args, model_type, model_name, lr, device):
    train_loader, dev_loader, test_loader, lang, tokenizer, _, _, _ = make_model_dataloaders(
        args,
        model_type,
        model_name,
        device,
    )

    print_model_info(model_type, model_name, train_loader, dev_loader, test_loader, lang)

    return run(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        lang=lang,
        tokenizer=tokenizer,
        device=device,
        model_type=model_type,
        model_name=model_name,
        output_dir=Path(args.output_dir) / model_type,
        runs=args.runs,
        seed=args.seed,
        lr=lr,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        n_epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
    )


# Rebuild the correct tokenizer and test loader from the metadata stored in a saved checkpoint.
def evaluate_saved_model(args, device):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when --eval is specified.")

    if args.model == "both":
        raise ValueError("When using --eval, choose exactly one model with --model bert or --model gpt2.")

    # The checkpoint determines the exact pretrained backbone used during training.
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_type = checkpoint["model_type"]
    model_name = checkpoint["model_name"]

    if args.model != model_type:
        raise ValueError(
            f"The selected model is '{args.model}', but the checkpoint contains '{model_type}'."
        )

    train_loader, dev_loader, test_loader, lang, tokenizer, _, _, _ = make_model_dataloaders(
        args,
        model_type,
        model_name,
        device,
    )

    print_model_info(model_type, model_name, train_loader, dev_loader, test_loader, lang)

    return eval_model_saved(
        checkpoint_path=args.checkpoint,
        test_loader=test_loader,
        lang=lang,
        tokenizer=tokenizer,
        device=device,
    )


def main():
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Evaluation-only mode skips all training and requires one explicit backbone.
    if args.eval:
        evaluate_saved_model(args, device)
        return

    # Without --eval, train the selected backbone or both backbones sequentially.
    if args.model in ["gpt2", "both"]:
        run_model(args, "gpt2", args.gpt2_model_name, args.gpt2_lr, device)

    if args.model in ["bert", "both"]:
        run_model(args, "bert", args.bert_model_name, args.bert_lr, device)


if __name__ == "__main__":
    main()
