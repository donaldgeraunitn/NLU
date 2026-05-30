import argparse
from pathlib import Path

import torch

from functions import *
from model import GPT2Baseline, GPT2Dropout, GPT2WeightTying, GPT2DropoutWeightTying
from utils import build_vocabulary, load_dataset, load_tokenizer, make_dataloaders


FIXED_MODEL_CONFIG = {
    "pos_emb_size": 1024,
    "d_model": 20,
    "n_heads": 1,
    "num_layers": 1,
    "ff_dim": 20,
    "dropout": 0.0,
}

DEFAULT_LEARNING_RATES = [0.001, 0.002, 0.003, 0.005]
DEFAULT_D_MODEL_VALUES = [20, 40, 80, 160]
DEFAULT_N_HEADS_VALUES = [1, 2, 4, 8]
DEFAULT_NUM_LAYERS_VALUES = [1, 2, 3, 4]
DEFAULT_FF_DIM_VALUES = [20, 40, 80, 160, 320]
DEFAULT_DROPOUT_VALUES = [0.1, 0.2, 0.3, 0.5]

MODEL_REGISTRY = {
    "GPT2Baseline": GPT2Baseline,
    "GPT2Dropout": GPT2Dropout,
    "GPT2WeightTying": GPT2WeightTying,
    "GPT2DropoutWeightTying": GPT2DropoutWeightTying,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Part 1.A - GPT-2 experiments")

    parser.add_argument(
        "--experiment",
        type=str,
        default="baseline_lr",
        choices=["baseline_lr", "greedy_hparams", "final_lr", "dropout", "weight_tying"],
    )

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="openai-community/gpt2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--learning_rates", type=float, nargs="+", default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--learning_rate", type=float, default=None)

    parser.add_argument("--d_model_values", type=int, nargs="+", default=DEFAULT_D_MODEL_VALUES)
    parser.add_argument("--n_heads_values", type=int, nargs="+", default=DEFAULT_N_HEADS_VALUES)
    parser.add_argument("--num_layers_values", type=int, nargs="+", default=DEFAULT_NUM_LAYERS_VALUES)
    parser.add_argument("--ff_dim_values", type=int, nargs="+", default=DEFAULT_FF_DIM_VALUES)
    parser.add_argument("--dropout_values", type=float, nargs="+", default=DEFAULT_DROPOUT_VALUES)

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--baseline_report", type=str, default="outputs/baseline_lr_tuning/report.json")
    parser.add_argument("--hparams_report", type=str, default="outputs/greedy_hparams/report.json")
    parser.add_argument("--final_lr_report", type=str, default="outputs/final_lr_readjustment/report.json")
    parser.add_argument("--dropout_report", type=str, default="outputs/dropout_tuning/report.json")

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args()


def get_device(args):
    if args.device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def get_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir

    return {
        "baseline_lr": "outputs/baseline_lr_tuning",
        "greedy_hparams": "outputs/greedy_hparams",
        "final_lr": "outputs/final_lr_readjustment",
        "dropout": "outputs/dropout_tuning",
        "weight_tying": "outputs/weight_tying",
    }.get(args.experiment, "outputs")


def get_model_config_from_report(report, default_config):
    if report is None:
        return dict(default_config)
    if "best" in report and "model_config" in report["best"]:
        return dict(report["best"]["model_config"])
    if "final_model_config" in report:
        return dict(report["final_model_config"])
    if "fixed_model_hyperparameters" in report:
        return dict(report["fixed_model_hyperparameters"])
    return dict(default_config)


def get_learning_rate_from_report(report):
    if report is None:
        return None
    if "best" in report and "learning_rate" in report["best"]:
        return report["best"]["learning_rate"]
    if "learning_rate" in report:
        return report["learning_rate"]
    if "best_learning_rate" in report:
        return report["best_learning_rate"]
    return None


def get_dev_ppl_from_report(report):
    if report is None:
        return None
    if "best" in report and "dev_ppl" in report["best"]:
        return report["best"]["dev_ppl"]
    if "best_dev_ppl" in report:
        return report["best_dev_ppl"]
    if "dev_ppl" in report:
        return report["dev_ppl"]
    return None


def get_learning_rate_or_raise(args, *reports):
    if args.learning_rate is not None:
        return args.learning_rate

    for report in reports:
        learning_rate = get_learning_rate_from_report(report)
        if learning_rate is not None:
            return learning_rate

    raise ValueError("No learning rate found. Run baseline_lr first or pass --learning_rate manually.")


def print_dataset_info(train_raw, dev_raw, test_raw, vocabulary):
    print(f"Train sentences: {len(train_raw)}")
    print(f"Dev sentences:   {len(dev_raw)}")
    print(f"Test sentences:  {len(test_raw)}")
    print(f"Vocabulary size: {len(vocabulary)}")


def tuning_kwargs(args):
    return {
        "seed": args.seed,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "n_epochs": args.n_epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
    }


def run_greedy_experiment(args, tokenizer, device, output_dir):
    baseline_report = load_report(args.baseline_report)
    start_model_config = get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG)

    current_config = dict(start_model_config)
    current_config["dropout"] = 0.0

    learning_rate = get_learning_rate_or_raise(args, baseline_report)

    print(f"Starting greedy hyperparameter tuning from config: {current_config}")
    print(f"Using learning rate: {learning_rate}")

    search_space = [
        ("d_model", args.d_model_values),
        ("n_heads", args.n_heads_values),
        ("num_layers", args.num_layers_values),
        ("ff_dim", args.ff_dim_values),
    ]

    steps = []
    final_best = None

    for parameter_name, candidate_values in search_space:
        print("\n" + "=" * 70)
        print(f"Greedy tuning step: {parameter_name}")
        print(f"Current fixed config: {current_config}")
        print("=" * 70)

        results, best = tune(
            parameter_name=parameter_name,
            candidate_values=candidate_values,
            model_cls=GPT2Baseline,
            base_model_config=current_config,
            tokenizer=tokenizer,
            make_dataloaders_fn=make_dataloaders,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir,
            **tuning_kwargs(args),
        )

        current_config = dict(best["model_config"])
        final_best = best

        step_plot = Path(output_dir) / f"tuning_{parameter_name}.png"
        plot_tuning_results(results, step_plot, parameter_name, title=f"Greedy tuning {parameter_name}")

        steps.append(
            {
                "param_name": parameter_name,
                "tested_values": list(candidate_values),
                "results": results,
                "best": {
                    parameter_name: current_config[parameter_name],
                    "dev_ppl": float(best["history"]["best_ppl"]),
                    "model_config": dict(current_config),
                    "plot": str(step_plot),
                },
            }
        )

        print(f"Best {parameter_name}: {current_config[parameter_name]}")
        print(f"Best dev PPL for this step: {best['history']['best_ppl']:.2f}")

    checkpoint_path, test_ppl, test_loss = save_best_model(
        output_dir=output_dir,
        model_cls=GPT2Baseline,
        model_config=current_config,
        model_state=final_best["model_state"],
        tokenizer=tokenizer,
        make_dataloaders_fn=make_dataloaders,
        device=device,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
        checkpoint_extra={
            "learning_rate": float(learning_rate),
            "dev_ppl": float(final_best["history"]["best_ppl"]),
        },
    )

    report = {
        "experiment_name": "greedy_hyperparameter_tuning",
        "seed": args.seed,
        "device": str(device),
        "model_class": GPT2Baseline.__name__,
        "learning_rate": float(learning_rate),
        "start_model_config": dict(start_model_config),
        "final_model_config": dict(current_config),
        "fixed_training_hyperparameters": fixed_training_hyperparameters(
            args.train_batch_size,
            args.eval_batch_size,
            args.n_epochs,
            args.patience,
            args.weight_decay,
            args.grad_clip,
        ),
        "steps": steps,
        "best": {
            "learning_rate": float(learning_rate),
            "best_epoch": int(final_best["history"]["best_epoch"]),
            "dev_ppl": float(final_best["history"]["best_ppl"]),
            "test_loss": float(test_loss),
            "test_ppl": float(test_ppl),
            "model_config": dict(current_config),
            "model_path": str(checkpoint_path),
        },
    }

    save_report(report, Path(output_dir) / "report.json")
    return report


def evaluate_checkpoint(args, tokenizer, device):
    if args.checkpoint is None:
        raise ValueError("Use --checkpoint PATH together with --eval.")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model_config = checkpoint.get("model_config", FIXED_MODEL_CONFIG)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model_class_name = checkpoint.get("model_class", "GPT2Baseline")
    model_cls = MODEL_REGISTRY.get(model_class_name, GPT2Baseline)

    _, test_ppl, test_loss = eval_model_saved(
        model_cls=model_cls,
        model_config=model_config,
        model_state=model_state,
        tokenizer=tokenizer,
        make_dataloaders_fn=make_dataloaders,
        device=device,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model class: {model_class_name}")
    print(f"Model config: {model_config}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test PPL:  {test_ppl:.2f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args)
    output_dir = get_output_dir(args)

    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")
    print(f"Experiment: {args.experiment}")
    print(f"Output dir: {output_dir}")

    train_raw, dev_raw, test_raw = load_dataset(eos_token="<eos>")
    tokenizer = load_tokenizer(args.tokenizer_name)
    vocabulary = build_vocabulary(tokenizer)
    print_dataset_info(train_raw, dev_raw, test_raw, vocabulary)

    if args.eval:
        evaluate_checkpoint(args, tokenizer, device)
        return

    common = {
        "tokenizer": tokenizer,
        "make_dataloaders_fn": make_dataloaders,
        "device": device,
        "output_dir": output_dir,
        **tuning_kwargs(args),
    }

    if args.experiment == "baseline_lr":
        print(f"Fixed model hyperparameters: {FIXED_MODEL_CONFIG}")
        print(f"Learning rates to test: {args.learning_rates}")

        report = run(
            parameter_name="learning_rate",
            candidate_values=args.learning_rates,
            model_cls=GPT2Baseline,
            model_config=FIXED_MODEL_CONFIG,
            learning_rate=args.learning_rates[0],
            experiment_name="baseline_lr_tuning",
            **common,
        )

    elif args.experiment == "greedy_hparams":
        report = run_greedy_experiment(args, tokenizer, device, output_dir)

    elif args.experiment == "final_lr":
        baseline_report = load_report(args.baseline_report)
        hparams_report = load_report(args.hparams_report)

        model_config = get_model_config_from_report(
            hparams_report,
            get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG),
        )
        model_config["dropout"] = 0.0

        print(f"Readjusting learning rate for final architecture: {model_config}")
        print(f"Learning rates to test: {args.learning_rates}")

        report = run(
            parameter_name="learning_rate",
            candidate_values=args.learning_rates,
            model_cls=GPT2Baseline,
            model_config=model_config,
            learning_rate=args.learning_rates[0],
            experiment_name="final_lr_readjustment",
            **common,
        )

    elif args.experiment == "dropout":
        baseline_report = load_report(args.baseline_report)
        hparams_report = load_report(args.hparams_report)
        final_lr_report = load_report(args.final_lr_report)

        model_config = get_model_config_from_report(
            hparams_report,
            get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG),
        )
        model_config["dropout"] = 0.0

        learning_rate = get_learning_rate_or_raise(args, final_lr_report, hparams_report, baseline_report)

        print(f"Starting dropout tuning from config: {model_config}")
        print(f"Using learning rate: {learning_rate}")
        print(f"Dropout values to test: {args.dropout_values}")

        report = run(
            parameter_name="dropout",
            candidate_values=args.dropout_values,
            model_cls=GPT2Dropout,
            model_config=model_config,
            learning_rate=learning_rate,
            experiment_name="dropout_tuning",
            **common,
        )

    elif args.experiment == "weight_tying":
        baseline_report = load_report(args.baseline_report)
        hparams_report = load_report(args.hparams_report)
        final_lr_report = load_report(args.final_lr_report)
        dropout_report = load_report(args.dropout_report)

        model_config = get_model_config_from_report(
            final_lr_report,
            get_model_config_from_report(
                hparams_report,
                get_model_config_from_report(baseline_report, FIXED_MODEL_CONFIG),
            ),
        )
        model_config["dropout"] = 0.0
        model_cls = GPT2WeightTying

        final_lr_ppl = get_dev_ppl_from_report(final_lr_report)
        dropout_ppl = get_dev_ppl_from_report(dropout_report)

        if dropout_report is not None and dropout_ppl is not None and (final_lr_ppl is None or dropout_ppl <= final_lr_ppl):
            model_config = get_model_config_from_report(dropout_report, model_config)
            model_cls = GPT2DropoutWeightTying
            print("Using the dropout configuration because it improved or matched the previous best dev PPL.")
        else:
            print("Using the non-dropout configuration for weight tying.")

        print(f"Starting weight tying from config: {model_config}")
        print(f"Learning rates to test: {args.learning_rates}")

        report = run(
            parameter_name="learning_rate",
            candidate_values=args.learning_rates,
            model_cls=model_cls,
            model_config=model_config,
            learning_rate=args.learning_rates[0],
            experiment_name="weight_tying",
            **common,
        )

    print("\nFinal results")
    print("-" * 40)
    print(f"Dev PPL:      {report['best']['dev_ppl']:.2f}")
    print(f"Test PPL:     {report['best']['test_ppl']:.2f}")
    print(f"Config:       {report['best']['model_config']}")
    print(f"Saved model:  {report['best']['model_path']}")
    print(f"Saved report: {Path(output_dir) / 'report.json'}")


if __name__ == "__main__":
    main()
