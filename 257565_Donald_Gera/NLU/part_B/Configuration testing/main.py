import argparse
import json
from pathlib import Path

import torch

from functions import run_experiment
from utils import make_dataloaders


BERT_CONFIGS = [
    {
        "name": "paper",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 5e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adam",
    },
    {
        "name": "adamw_lr_2e-5",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 2e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "adamw_lr_3e-5",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 3e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "adamw_lr_5e-5",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 5e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "adamw_dropout_0.0",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 5e-5,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "adamw_dropout_0.2",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 5e-5,
        "dropout": 0.2,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "adamw_weight_decay_0.01",
        "model_name": "google-bert/bert-base-uncased",
        "lr": 5e-5,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "optimizer": "adamw",
    },
]

GPT2_CONFIGS = [
    {
        "name": "baseline_lr_5e-5",
        "model_name": "openai-community/gpt2",
        "lr": 5e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "lr_1e-5",
        "model_name": "openai-community/gpt2",
        "lr": 1e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "lr_2e-5",
        "model_name": "openai-community/gpt2",
        "lr": 2e-5,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "lr_1e-4",
        "model_name": "openai-community/gpt2",
        "lr": 1e-4,
        "dropout": 0.1,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "dropout_0.0",
        "model_name": "openai-community/gpt2",
        "lr": 5e-5,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "dropout_0.2",
        "model_name": "openai-community/gpt2",
        "lr": 5e-5,
        "dropout": 0.2,
        "weight_decay": 0.0,
        "optimizer": "adamw",
    },
    {
        "name": "weight_decay_0.01",
        "model_name": "openai-community/gpt2",
        "lr": 5e-5,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "optimizer": "adamw",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2", "bert", "both"], required=True)
    parser.add_argument("--config", default="all")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--dataset-dir", default="dataset/ATIS")
    parser.add_argument("--output-dir", default="outputs")

    # Used only with --config custom.
    parser.add_argument("--gpt2-model-name", default="openai-community/gpt2")
    parser.add_argument("--bert-model-name", default="google-bert/bert-base-uncased")
    parser.add_argument("--gpt2-lr", type=float, default=5e-5)
    parser.add_argument("--bert-lr", type=float, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--dev-size", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def get_model_types(model):
    if model == "both":
        return ["gpt2", "bert"]
    return [model]


def get_configs(args, model_type):
    if args.config == "custom":
        return [
            {
                "name": "custom",
                "model_name": args.gpt2_model_name if model_type == "gpt2" else args.bert_model_name,
                "lr": args.gpt2_lr if model_type == "gpt2" else args.bert_lr,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "optimizer": args.optimizer,
            }
        ]

    configs = GPT2_CONFIGS if model_type == "gpt2" else BERT_CONFIGS
    if args.config == "all":
        return configs

    selected = [config for config in configs if config["name"] == args.config]
    if not selected:
        available = ", ".join(config["name"] for config in configs)
        raise ValueError(
            f"Unknown {model_type} configuration: {args.config}. "
            f"Available configurations: {available}, custom"
        )
    return selected


def print_configs(model_type, configs):
    print(f"{model_type.upper()} configurations:")
    for config in configs:
        print(
            f"- {config['name']}: model={config['model_name']}, "
            f"optimizer={config['optimizer']}, lr={config['lr']}, "
            f"dropout={config['dropout']}, weight_decay={config['weight_decay']}"
        )


def run_configuration(args, model_type, config, device):
    train_loader, dev_loader, test_loader, lang, tokenizer = make_dataloaders(
        device=device,
        model_type=model_type,
        model_name=config["model_name"],
        dataset_dir=args.dataset_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        data_seed=args.data_seed,
        dev_size=args.dev_size,
        max_length=args.max_length,
    )

    print("=" * 89)
    print("Model:", model_type)
    print("Configuration:", config["name"])
    print("Checkpoint:", config["model_name"])
    print("Optimizer:", config["optimizer"])
    print("Learning rate:", config["lr"])
    print("Dropout:", config["dropout"])
    print("Weight decay:", config["weight_decay"])
    print("Train samples:", len(train_loader.dataset))
    print("Dev samples:", len(dev_loader.dataset))
    print("Test samples:", len(test_loader.dataset))
    print("Slot labels:", len(lang.slot2id))
    print("Intent labels:", len(lang.intent2id))
    print("=" * 89)

    return run_experiment(
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        lang=lang,
        tokenizer=tokenizer,
        device=device,
        model_type=model_type,
        model_name=config["model_name"],
        output_dir=Path(args.output_dir) / model_type / config["name"],
        runs=args.runs,
        seed=args.seed,
        lr=config["lr"],
        dropout=config["dropout"],
        weight_decay=config["weight_decay"],
        optimizer_name=config["optimizer"],
        n_epochs=args.epochs,
        patience=args.patience,
        eval_every=args.eval_every,
    )


def main():
    args = parse_args()
    model_types = get_model_types(args.model)

    if args.list_configs:
        for model_type in model_types:
            print_configs(model_type, get_configs(args, model_type))
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for model_type in model_types:
        summaries = []
        for config in get_configs(args, model_type):
            summaries.append(run_configuration(args, model_type, config, device))

        model_output_dir = Path(args.output_dir) / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        with open(
            model_output_dir / f"{model_type}_all_results.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summaries, f, indent=4)


if __name__ == "__main__":
    main()
