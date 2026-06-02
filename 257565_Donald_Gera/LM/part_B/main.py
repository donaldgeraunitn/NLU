import argparse
from pathlib import Path

import torch

from functions import evaluate_checkpoint, run, set_seed
from utils import build_vocabulary, load_dataset, load_tokenizer, make_dataloaders


# Default LoRA search space used by --experiments.
DEFAULT_RANK_VALUES = [2, 4, 8, 16]
DEFAULT_ALPHA_VALUES = [8, 16, 32]


# Keep training, sweep, and saved-checkpoint evaluation options in one entry point.
def parse_args():
    parser = argparse.ArgumentParser(description="Part B - LoRA")

    parser.add_argument("--experiments", action="store_true", help="Run all rank and alpha configurations.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--rank_values", type=int, nargs="+", default=DEFAULT_RANK_VALUES)
    parser.add_argument("--alpha_values", type=int, nargs="+", default=DEFAULT_ALPHA_VALUES)

    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--part_a_ppl", type=float, default=None)

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args()


# Use the first CUDA device when available, while still supporting CPU execution.
def get_device(args):
    if args.device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def print_dataset_info(train_raw, dev_raw, test_raw, vocabulary):
    print(f"Train sentences: {len(train_raw)}")
    print(f"Dev sentences:   {len(dev_raw)}")
    print(f"Test sentences:  {len(test_raw)}")
    print(f"Vocabulary size: {len(vocabulary)}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args)

    print(f"Using device: {device}")
    print(f"Using seed: {args.seed}")

    # During evaluation, load the model name stored in the checkpoint so the
    # tokenizer matches the pretrained backbone used during training.
    tokenizer_name = args.model_name
    if args.eval:
        if args.checkpoint is None:
            raise ValueError("Use --checkpoint PATH together with --eval.")
        checkpoint_preview = torch.load(args.checkpoint, map_location="cpu")
        tokenizer_name = checkpoint_preview["model_config"]["model_name"]

    # The dataset path is fixed in utils.py; commands must be run from this part directory.
    tokenizer = load_tokenizer(tokenizer_name)
    train_raw, dev_raw, test_raw = load_dataset(eos_token="<eos>")
    vocabulary = build_vocabulary(tokenizer)
    print_dataset_info(train_raw, dev_raw, test_raw, vocabulary)

    # Evaluation-only mode rebuilds the saved LoRA model and skips training entirely.
    if args.eval:
        checkpoint, test_ppl, test_loss = evaluate_checkpoint(checkpoint_path=args.checkpoint, tokenizer=tokenizer, make_dataloaders_fn=make_dataloaders, device=device, eval_batch_size=args.eval_batch_size)

        print(f"Loaded model checkpoint: {args.checkpoint}")
        print(f"Model config: {checkpoint['model_config']}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test PPL:  {test_ppl:.2f}")
        return

    # --experiments evaluates the Cartesian product of rank and alpha values.
    # Without it, only the single --rank/--alpha configuration is trained.
    if args.experiments:
        rank_values = args.rank_values
        alpha_values = args.alpha_values
        experiment_name = "all configurations"
    else:
        rank_values = [args.rank]
        alpha_values = [args.alpha]
        experiment_name = "single configuration"

    print(f"Experiment: {experiment_name}")
    print(f"Ranks to test: {rank_values}")
    print(f"Alpha values to test: {alpha_values}")
    print(f"Output dir: {args.output_dir}")

    # run() trains the requested configurations, selects the best one on
    # development perplexity, and saves its checkpoint under output_dir.
    report = run(model_name=args.model_name, rank_values=rank_values, alpha_values=alpha_values, tokenizer=tokenizer, make_dataloaders_fn=make_dataloaders, device=device, output_dir=args.output_dir, learning_rate=args.learning_rate, seed=args.seed, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, n_epochs=args.n_epochs, patience=args.patience, weight_decay=args.weight_decay, part_a_ppl=args.part_a_ppl)

    best = report["best"]

    print("\nFinal results")
    print("-" * 50)
    print(f"Best rank:       {best['rank']}")
    print(f"Best alpha:      {best['alpha']}")
    print(f"Dev PPL:         {best['dev_ppl']:.2f}")
    print(f"Test PPL:        {best['test_ppl']:.2f}")
    print(f"Saved model:     {best['model_path']}")
    print(f"Saved report:    {Path(args.output_dir) / 'report.json'}")
    print(f"PPL below 250:   {report['requirements']['ppl_below_250']}")
    if args.part_a_ppl is not None:
        print(f"Part 1.A PPL:    {args.part_a_ppl:.2f}")
        print(f"Below Part 1.A:  {report['requirements']['ppl_lower_than_part_a']}")


if __name__ == "__main__":
    main()
