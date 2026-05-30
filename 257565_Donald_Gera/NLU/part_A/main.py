import argparse
from pathlib import Path

import torch

from functions import get_intent_accuracy, get_slot_f1, load_report, save_report, set_seed, tune
from model import GPT2
from utils import make_dataloaders, make_datasets


FIXED_MODEL_CONFIG = {
    "pos_emb_size": 1024,
    "d_model": 20,
    "n_heads": 1,
    "num_layers": 1,
    "ff_dim": 20,
    "dropout": 0.0,
}

DEFAULT_LEARNING_RATES = [0.0005, 0.001, 0.002, 0.003, 0.005]
DEFAULT_D_MODEL_VALUES = [20, 40, 80, 160]
DEFAULT_N_HEADS_VALUES = [1, 2, 4, 8]
DEFAULT_NUM_LAYERS_VALUES = [1, 2, 3, 4]
DEFAULT_FF_DIM_VALUES = [20, 40, 80, 160, 320]
DEFAULT_DROPOUT_VALUES = [0.1, 0.2, 0.3, 0.5]


def parse_args():
    parser = argparse.ArgumentParser(description="Part 2.A - ATIS intent classification and slot filling")

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline_lr",
        choices=["baseline_lr", "greedy_hparams", "final_lr", "dropout", "final_runs", "custom"],
    )

    parser.add_argument("--dataset_dir", type=str, default="dataset/ATIS")
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--runs_per_config", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="slot_f1",
        choices=["slot_f1", "intent_acc", "average"],
    )

    parser.add_argument("--learning_rates", type=float, nargs="+", default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--learning_rate", type=float, default=None)

    parser.add_argument("--d_model_values", type=int, nargs="+", default=DEFAULT_D_MODEL_VALUES)
    parser.add_argument("--n_heads_values", type=int, nargs="+", default=DEFAULT_N_HEADS_VALUES)
    parser.add_argument("--num_layers_values", type=int, nargs="+", default=DEFAULT_NUM_LAYERS_VALUES)
    parser.add_argument("--ff_dim_values", type=int, nargs="+", default=DEFAULT_FF_DIM_VALUES)
    parser.add_argument("--dropout_values", type=float, nargs="+", default=DEFAULT_DROPOUT_VALUES)

    # Values used by --experiment custom.
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    parser.add_argument("--baseline_report", type=str, default="outputs/part2a/baseline_lr_tuning/report.json")
    parser.add_argument("--hparams_report", type=str, default="outputs/part2a/greedy_hparams/report.json")
    parser.add_argument("--final_lr_report", type=str, default="outputs/part2a/final_lr_readjustment/report.json")
    parser.add_argument("--dropout_report", type=str, default="outputs/part2a/dropout_tuning/report.json")

    return parser.parse_args()


def get_device(args):
    if args.device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def get_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir

    mapping = {
        "baseline_lr": "outputs/part2a/baseline_lr_tuning",
        "greedy_hparams": "outputs/part2a/greedy_hparams",
        "final_lr": "outputs/part2a/final_lr_readjustment",
        "dropout": "outputs/part2a/dropout_tuning",
        "final_runs": "outputs/part2a/final_runs",
        "custom": "outputs/part2a/custom",
    }
    return mapping[args.experiment]


def get_model_config_from_report(report, default_config):
    if report is not None and "best" in report and "model_config" in report["best"]:
        return dict(report["best"]["model_config"])
    return dict(default_config)


def get_learning_rate_from_report(report):
    if report is not None and "best" in report:
        return report["best"].get("learning_rate")
    return None


def get_learning_rate_or_raise(args, *reports):
    if args.learning_rate is not None:
        return args.learning_rate

    for report in reports:
        learning_rate = get_learning_rate_from_report(report)
        if learning_rate is not None:
            return learning_rate

    raise ValueError("No learning rate found. Run baseline_lr first or pass --learning_rate manually.")


def print_dataset_info(dataset_dir, data_seed):
    _, _, _, lang, train_raw, dev_raw, test_raw = make_datasets(
        dataset_dir=dataset_dir,
        seed=data_seed,
    )
    print(f"Train sentences: {len(train_raw)}")
    print(f"Dev sentences:   {len(dev_raw)}")
    print(f"Test sentences:  {len(test_raw)}")
    print(f"Vocabulary size: {len(lang.word2id)}")
    print(f"Slots:           {len(lang.id2slot)}")
    print(f"Intents:         {len(lang.intent2id)}")


def common_tune_args(args, device):
    return {
        "model_cls": GPT2,
        "make_dataloaders_fn": make_dataloaders,
        "device": device,
        "runs_per_config": args.runs_per_config,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "dataset_dir": args.dataset_dir,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "n_epochs": args.n_epochs,
        "patience": args.patience,
        "eval_every": args.eval_every,
        "weight_decay": args.weight_decay,
        "selection_metric": args.selection_metric,
    }


def print_final_summary(report):
    best = report["best"]
    dev = best["dev"]

    print("\nFinal results")
    print("-" * 40)
    print(f"Dev slot F1:     {dev['slot_f1']['mean']:.4f} +- {dev['slot_f1']['std']:.4f}")
    print(f"Dev intent acc:  {dev['intent_acc']['mean']:.4f} +- {dev['intent_acc']['std']:.4f}")

    if "test" in best:
        test = best["test"]
        print(f"Test slot F1:    {test['slot_f1']['mean']:.4f} +- {test['slot_f1']['std']:.4f}")
        print(f"Test intent acc: {test['intent_acc']['mean']:.4f} +- {test['intent_acc']['std']:.4f}")

    print(f"Config:          {best['model_config']}")
    print(f"Learning rate:   {best['learning_rate']}")
    print(f"Saved model:     {best['best_individual_checkpoint']}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args)
    output_dir = Path(get_output_dir(args))
    tune_args = common_tune_args(args, device)

    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")
    print(f"Using fixed data seed: {args.data_seed}")
    print(f"Experiment: {args.experiment}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Output dir: {output_dir}")
    print_dataset_info(args.dataset_dir, args.data_seed)

    if args.experiment == "baseline_lr":
        report = tune(
            parameter_name="learning_rate",
            candidate_values=args.learning_rates,
            base_model_config=FIXED_MODEL_CONFIG,
            base_learning_rate=None,
            output_dir=output_dir,
            experiment_name="baseline_lr_tuning",
            evaluate_test=False,
            **tune_args,
        )

    elif args.experiment == "greedy_hparams":
        baseline_report = load_report(args.baseline_report)
        current_config = get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG)
        current_config["dropout"] = 0.0
        learning_rate = get_learning_rate_or_raise(args, baseline_report)

        search_space = [
            ("d_model", args.d_model_values),
            ("n_heads", args.n_heads_values),
            ("num_layers", args.num_layers_values),
            ("ff_dim", args.ff_dim_values),
        ]

        steps = []
        for parameter_name, values in search_space:
            step_report = tune(
                parameter_name=parameter_name,
                candidate_values=values,
                base_model_config=current_config,
                base_learning_rate=learning_rate,
                output_dir=output_dir / parameter_name,
                experiment_name=f"greedy_{parameter_name}",
                evaluate_test=False,
                **tune_args,
            )
            current_config = dict(step_report["best"]["model_config"])
            steps.append(
                {
                    "parameter_name": parameter_name,
                    "report_path": str(output_dir / parameter_name / "report.json"),
                    "best": step_report["best"],
                }
            )

        report = {
            "experiment_name": "greedy_hyperparameter_tuning",
            "selection_metric": args.selection_metric,
            "runs_per_config": args.runs_per_config,
            "learning_rate": float(learning_rate),
            "steps": steps,
            "best": {
                "model_config": dict(current_config),
                "learning_rate": float(learning_rate),
                "dev": steps[-1]["best"]["dev"],
                "best_individual_checkpoint": steps[-1]["best"]["best_individual_checkpoint"],
            },
        }
        save_report(report, output_dir / "report.json")

    elif args.experiment == "final_lr":
        baseline_report = load_report(args.baseline_report)
        hparams_report = load_report(args.hparams_report)
        model_config = get_model_config_from_report(
            hparams_report,
            get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG),
        )
        model_config["dropout"] = 0.0

        report = tune(
            parameter_name="learning_rate",
            candidate_values=args.learning_rates,
            base_model_config=model_config,
            base_learning_rate=None,
            output_dir=output_dir,
            experiment_name="final_lr_readjustment",
            evaluate_test=False,
            **tune_args,
        )

    elif args.experiment == "dropout":
        baseline_report = load_report(args.baseline_report)
        hparams_report = load_report(args.hparams_report)
        final_lr_report = load_report(args.final_lr_report)

        model_config = get_model_config_from_report(
            final_lr_report,
            get_model_config_from_report(
                hparams_report,
                get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG),
            ),
        )
        model_config["dropout"] = 0.0
        learning_rate = get_learning_rate_or_raise(args, final_lr_report, hparams_report, baseline_report)

        report = tune(
            parameter_name="dropout",
            candidate_values=args.dropout_values,
            base_model_config=model_config,
            base_learning_rate=learning_rate,
            output_dir=output_dir,
            experiment_name="dropout_tuning",
            evaluate_test=True,
            **tune_args,
        )

    elif args.experiment == "final_runs":
        dropout_report = load_report(args.dropout_report)
        final_lr_report = load_report(args.final_lr_report)
        hparams_report = load_report(args.hparams_report)
        baseline_report = load_report(args.baseline_report)

        source_report = dropout_report or final_lr_report or hparams_report or baseline_report
        model_config = get_model_config_from_report(source_report, FIXED_MODEL_CONFIG)
        learning_rate = get_learning_rate_or_raise(
            args,
            dropout_report,
            final_lr_report,
            hparams_report,
            baseline_report,
        )

        report = tune(
            parameter_name=None,
            candidate_values=[None],
            base_model_config=model_config,
            base_learning_rate=learning_rate,
            output_dir=output_dir,
            experiment_name="final_runs",
            evaluate_test=True,
            **tune_args,
        )

    else:  # custom
        model_config = dict(FIXED_MODEL_CONFIG)
        for key in ["d_model", "n_heads", "num_layers", "ff_dim", "dropout"]:
            value = getattr(args, key)
            if value is not None:
                model_config[key] = value

        if args.learning_rate is None:
            raise ValueError("Use --learning_rate for a custom configuration.")

        report = tune(
            parameter_name=None,
            candidate_values=[None],
            base_model_config=model_config,
            base_learning_rate=args.learning_rate,
            output_dir=output_dir,
            experiment_name="custom",
            evaluate_test=True,
            **tune_args,
        )

    print_final_summary(report)
    print(f"Saved report:    {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
