import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from model import GPT2_LoRA, param_stats


# Seed Python, NumPy, and PyTorch to make comparisons between LoRA
# configurations as reproducible as possible.
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Perplexity is the exponential of the average language-model loss.
def get_ppl(loss):
    if loss >= 50:
        return float("inf")
    return float(math.exp(loss))


# Train for one epoch. GPT2LMHeadModel performs the causal label shift
# internally when labels are passed to its forward method.
def train_loop(data, optimizer, model, tokenizer):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, attention_mask, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        # We do not shift labels manually: GPT2LMHeadModel manages it internally.
        labels = input_ids.clone().detach()
        # Ignore only padded positions. The pad token and EOS token can share the
        # same id in GPT-2, so the attention mask is safer than comparing token ids.
        labels[attention_mask == 0] = -100
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Weight each batch loss by its number of valid tokens so batches
        # with different amounts of padding contribute correctly.
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            pbar.set_postfix(loss=(sum(loss_array)/sum(number_of_tokens)).item())

    return sum(loss_array)/sum(number_of_tokens)


# Evaluate without gradient tracking using the same masked LM loss as training.
def eval_loop(data, model, tokenizer):
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        pbar = tqdm(data, desc="Evaluating:", unit="batch", total=len(data))

        for input_ids, attention_mask, n_tokens in pbar:
            labels = input_ids.clone().detach()
            labels[attention_mask == 0] = -100
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss = sum(loss_array)/sum(number_of_tokens)
    return get_ppl(loss), loss


# Store a detached CPU copy so the best state is independent of later updates
# and can be saved without keeping GPU tensors alive.
def model_state_on_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train(
    model_name,
    rank,
    alpha,
    tokenizer,
    make_dataloaders_fn,
    learning_rate,
    device,
    seed=42,
    train_batch_size=8,
    eval_batch_size=16,
    n_epochs=5,
    patience=3,
    weight_decay=0.01,
):
    set_seed(seed)

    train_loader, dev_loader, _ = make_dataloaders_fn(tokenizer=tokenizer, device=device, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, seed=seed)

    # Load pretrained GPT-2, inject LoRA adapters, and disable cache because
    # cached activations are unnecessary during full-sequence fine-tuning.
    model = GPT2_LoRA.from_pretrained(model_name, rank=rank, alpha=alpha).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # GPT2_LoRA freezes the backbone, so AdamW receives adapter parameters only.
    total_parameters, trainable_parameters = param_stats(model)
    optimizer = optim.AdamW(filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    losses_train = []
    losses_dev = []
    sampled_epochs = []

    best_ppl = math.inf
    best_epoch = -1
    best_model_state = None
    remaining_patience = patience

    pbar = tqdm(range(1, n_epochs + 1), desc="Epochs")

    for epoch in pbar:
        loss_train = train_loop(train_loader, optimizer, model, tokenizer)
        ppl_dev, loss_dev = eval_loop(dev_loader, model, tokenizer)

        sampled_epochs.append(epoch)
        losses_train.append(float(loss_train))
        losses_dev.append(float(loss_dev))

        pbar.set_description("PPL: %f" % ppl_dev)
        print(f"epoch={epoch:03d} | train_loss={loss_train:.4f} | dev_loss={loss_dev:.4f} | dev_ppl={ppl_dev:.2f}")

        # Select checkpoints only on development perplexity. The test split
        # is not used to decide when to stop training.
        if best_model_state is None or ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_epoch = epoch
            best_model_state = model_state_on_cpu(model)
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience <= 0:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Restore the best development-set state before test evaluation.
    model.load_state_dict(best_model_state)

    history = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
        "best_epoch": int(best_epoch),
        "best_ppl": float(best_ppl),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "trainable_percentage": float(100.0 * trainable_parameters / total_parameters),
    }

    return model, history, best_model_state


# Evaluate an already-built model on the held-out test split.
def evaluate(model, tokenizer, make_dataloaders_fn, device, eval_batch_size=16, seed=42):
    set_seed(seed)
    _, _, test_loader = make_dataloaders_fn(tokenizer=tokenizer, device=device, train_batch_size=eval_batch_size, eval_batch_size=eval_batch_size, seed=seed)
    return eval_loop(test_loader, model, tokenizer)


# Rebuild a saved LoRA configuration, restore its state, and evaluate it.
# This is the path used by main.py when --eval is specified.
def evaluate_checkpoint(checkpoint_path, tokenizer, make_dataloaders_fn, device, eval_batch_size=16):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]

    model = GPT2_LoRA.from_pretrained(model_config["model_name"], rank=model_config["rank"], alpha=model_config["alpha"]).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.load_state_dict(checkpoint["model_state_dict"])

    test_ppl, test_loss = evaluate(model=model, tokenizer=tokenizer, make_dataloaders_fn=make_dataloaders_fn, device=device, eval_batch_size=eval_batch_size, seed=checkpoint.get("seed", 42))
    return checkpoint, test_ppl, test_loss


def run(
    model_name,
    rank_values,
    alpha_values,
    tokenizer,
    make_dataloaders_fn,
    device,
    output_dir,
    learning_rate=2e-4,
    seed=42,
    train_batch_size=8,
    eval_batch_size=16,
    n_epochs=5,
    patience=3,
    weight_decay=0.01,
    part_a_ppl=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best = None
    checkpoint_path = output_dir / "best_model.pt"

    # Test every requested rank/alpha pair. Development perplexity chooses
    # the checkpoint that is retained as output_dir/best_model.pt.
    for rank in rank_values:
        for alpha in alpha_values:
            print("\n" + "=" * 80)
            print(f"Training configuration: rank={rank}, alpha={alpha}")
            print(f"LoRA scaling alpha / rank: {alpha / rank:.4f}")
            print("=" * 80)

            model, history, model_state = train(model_name=model_name, rank=rank, alpha=alpha, tokenizer=tokenizer, make_dataloaders_fn=make_dataloaders_fn, learning_rate=learning_rate, device=device, seed=seed, train_batch_size=train_batch_size, eval_batch_size=eval_batch_size, n_epochs=n_epochs, patience=patience, weight_decay=weight_decay)
            test_ppl, test_loss = evaluate(model=model, tokenizer=tokenizer, make_dataloaders_fn=make_dataloaders_fn, device=device, eval_batch_size=eval_batch_size, seed=seed)

            history_path = output_dir / f"history_rank_{rank}_alpha_{alpha}.png"
            plot_history(history, history_path, title=f"rank={rank}, alpha={alpha}")

            result = {
                "rank": int(rank),
                "alpha": int(alpha),
                "scaling": float(alpha / rank),
                "best_epoch": int(history["best_epoch"]),
                "dev_ppl": float(history["best_ppl"]),
                "test_loss": float(test_loss),
                "test_ppl": float(test_ppl),
                "trainable_parameters": int(history["trainable_parameters"]),
                "trainable_percentage": float(history["trainable_percentage"]),
                "history_plot": str(history_path),
            }
            results.append(result)

            print("\nConfiguration summary")
            print(f"Dev PPL:  {result['dev_ppl']:.2f}")
            print(f"Test PPL: {result['test_ppl']:.2f}")

            # Overwrite the saved checkpoint only when development PPL improves.
            if best is None or result["dev_ppl"] < best["dev_ppl"]:
                best = result
                torch.save(
                    {
                        "model_class": "GPT2_LoRA",
                        "model_config": {
                            "model_name": model_name,
                            "rank": int(rank),
                            "alpha": int(alpha),
                        },
                        "model_state_dict": model_state,
                        "seed": int(seed),
                        "selection_metric": "dev_ppl",
                        "dev_ppl": float(result["dev_ppl"]),
                        "test_ppl": float(result["test_ppl"]),
                    },
                    checkpoint_path,
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Record whether the selected model satisfies the exercise requirements.
    requirements = {
        "ppl_below_250": bool(best["test_ppl"] < 250.0),
        "part_a_ppl": None if part_a_ppl is None else float(part_a_ppl),
        "ppl_lower_than_part_a": None if part_a_ppl is None else bool(best["test_ppl"] < part_a_ppl),
    }

    tuning_plot_path = output_dir / "rank_alpha_tuning.png"
    plot_tuning_results(results, tuning_plot_path)

    report = {
        "experiment_name": "gpt2_lora_rank_alpha_tuning",
        "model_class": "GPT2_LoRA",
        "model_name": model_name,
        "seed": int(seed),
        "device": str(device),
        "rank_values": [int(value) for value in rank_values],
        "alpha_values": [int(value) for value in alpha_values],
        "fixed_training_hyperparameters": {
            "learning_rate": float(learning_rate),
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "n_epochs": int(n_epochs),
            "patience": int(patience),
            "weight_decay": float(weight_decay),
            "optimizer": "AdamW",
        },
        "results": results,
        "requirements": requirements,
        "best": {
            **best,
            "selection_metric": "dev_ppl",
            "model_path": str(checkpoint_path),
            "tuning_plot": str(tuning_plot_path),
        },
    }

    save_report(report, output_dir / "report.json")
    return report


# Save train/dev loss curves for one rank-alpha configuration.
def plot_history(history, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history["sampled_epochs"], history["losses_train"], label="train loss")
    plt.plot(history["sampled_epochs"], history["losses_dev"], label="dev loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Compare development perplexity across ranks, with one curve per alpha value.
def plot_tuning_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_values = sorted(set(result["alpha"] for result in results))

    plt.figure()
    for alpha in alpha_values:
        alpha_results = sorted([result for result in results if result["alpha"] == alpha], key=lambda result: result["rank"])
        plt.plot([result["rank"] for result in alpha_results], [result["dev_ppl"] for result in alpha_results], marker="o", label=f"alpha={alpha}")

    plt.xlabel("rank")
    plt.ylabel("dev perplexity")
    plt.title("GPT-2 LoRA rank and alpha tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Save both a machine-readable JSON report and a compact text summary.
def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"{report['experiment_name']} report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"model_name: {report['model_name']}\n")
        f.write(f"seed: {report['seed']}\n")
        f.write(f"device: {report['device']}\n")
        f.write(f"rank_values: {report['rank_values']}\n")
        f.write(f"alpha_values: {report['alpha_values']}\n")
        f.write(f"fixed_training_hyperparameters: {report['fixed_training_hyperparameters']}\n\n")

        f.write("Results:\n")
        for result in report["results"]:
            f.write(json.dumps(result, indent=4))
            f.write("\n")

        f.write("\nBest configuration:\n")
        f.write(json.dumps(report["best"], indent=4))
        f.write("\n")
