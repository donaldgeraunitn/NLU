import json
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


# Configure Python, NumPy, and PyTorch for reproducible experiment runs.
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


# Initialize the linear layers of each model trained from scratch.
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# Convert average cross-entropy loss to perplexity while avoiding overflow.
def get_ppl(loss):
    if loss >= 50:
        return float("inf")
    return float(math.exp(loss))


def train_loop(data, optimizer, criterion, model, grad_clip=None):
    """Run one training epoch and return the token-weighted average loss."""
    model.train()
    loss_array = []
    number_of_tokens = []

    pbar = tqdm(data, desc="Training:", unit="batch", total=len(data))

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()

        output = model(input_ids)
        # CrossEntropyLoss expects class scores before the sequence dimension.
        loss = criterion(output.permute(0, 2, 1), labels)

        # Weight batch losses by non-padding token count before averaging.
        loss_array.append(loss.item() * int(n_tokens.item()))
        number_of_tokens.append(int(n_tokens.item()))

        loss.backward()

        # Optional clipping limits unstable gradient updates in larger configurations.
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if i % 100 == 0:
            avg_loss = sum(loss_array) / max(sum(number_of_tokens), 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return sum(loss_array) / max(sum(number_of_tokens), 1)


def eval_loop(data, criterion, model, show_progress=True):
    """Evaluate a loader without gradients and return perplexity and average loss."""
    model.eval()
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        iterator = tqdm(data, desc="Evaluating:", unit="batch", total=len(data)) if show_progress else data

        for input_ids, labels, n_tokens in iterator:
            output = model(input_ids)
            loss = criterion(output.permute(0, 2, 1), labels)

            loss_array.append(loss.item() * int(n_tokens.item()))
            number_of_tokens.append(int(n_tokens.item()))

    loss = sum(loss_array) / max(sum(number_of_tokens), 1)
    return get_ppl(loss), loss


# Store a detached CPU copy so the best state is independent from later updates.
def model_state_on_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train(
    model_cls,
    model_config,
    tokenizer,
    make_dataloaders_fn,
    learning_rate,
    device,
    seed=42,
    train_batch_size=8,
    eval_batch_size=16,
    n_epochs=100,
    patience=3,
    min_delta=0.0,
    weight_decay=0.01,
    grad_clip=None,
):
    """
    Train one model from scratch.

    This is the only full training function. It creates the dataloaders, model,
    optimizer and loss, runs validation and early stopping, restores the best
    model state, and returns the relevant training information.
    """
    set_seed(seed)

    # Each candidate is trained from scratch with reproducibly shuffled batches.
    train_loader, dev_loader, _ = make_dataloaders_fn(
        tokenizer=tokenizer,
        device=device,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )

    # Build the selected model variant and initialize its linear projections.
    model = model_cls(vocab_size=len(tokenizer), **model_config).to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Padding positions must not affect the next-token prediction loss.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    losses_train = []
    losses_dev = []
    sampled_epochs = []

    best_ppl = math.inf
    best_epoch = -1
    best_model_state = None
    remaining_patience = patience

    pbar = tqdm(range(1, n_epochs + 1), desc="Epochs")

    for epoch in pbar:
        loss_train = train_loop(train_loader, optimizer, criterion, model, grad_clip=grad_clip)
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion, model)

        sampled_epochs.append(epoch)
        losses_train.append(float(loss_train))
        losses_dev.append(float(loss_dev))

        pbar.set_description("PPL: %f" % ppl_dev)

        if epoch % 10 == 0 or epoch == n_epochs:
            print ( f"epoch={epoch:03d} | train_loss={loss_train:.4f} | " f"dev_loss={loss_dev:.4f} | dev_ppl={ppl_dev:.2f}" )

        # Keep the state with the lowest validation perplexity and reset patience on improvement.
        if best_model_state is None or ppl_dev < best_ppl - min_delta:
            best_ppl = ppl_dev
            best_epoch = epoch
            best_model_state = model_state_on_cpu(model)
            remaining_patience = patience
        else:
            remaining_patience -= 1

        # Stop once validation perplexity has failed to improve for the configured patience.
        if remaining_patience <= 0:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Return the best validation state rather than the final epoch state.
    model.load_state_dict(best_model_state)

    history = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
        "best_epoch": best_epoch,
        "best_ppl": float(best_ppl),
    }

    return model, history, best_model_state


# Rebuild and evaluate a saved state using the same test pipeline used after training.
def eval_model_saved(
    model_cls,
    model_config,
    model_state,
    tokenizer,
    make_dataloaders_fn,
    device,
    eval_batch_size=16,
    seed=42,
    show_progress=True,
):
    set_seed(seed)

    # Only the test loader is used, but the shared helper keeps preprocessing identical.
    _, _, test_loader = make_dataloaders_fn(
        tokenizer=tokenizer,
        device=device,
        train_batch_size=eval_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )

    model = model_cls(vocab_size=len(tokenizer), **model_config).to(device)
    model.load_state_dict(model_state)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    test_ppl, test_loss = eval_loop(test_loader, criterion, model, show_progress=show_progress)

    return model, test_ppl, test_loss


# Train every candidate for one selected hyperparameter and retain the best state.
def tune(
    parameter_name,
    candidate_values,
    model_cls,
    base_model_config,
    tokenizer,
    make_dataloaders_fn,
    learning_rate,
    device,
    output_dir,
    seed=42,
    train_batch_size=8,
    eval_batch_size=16,
    n_epochs=100,
    patience=3,
    weight_decay=0.01,
    grad_clip=None,
    experiment_name=None,
):
    """
    Tune exactly one hyperparameter while keeping the remaining values fixed.

    This single function is reused for learning-rate search, every greedy
    architecture step, and dropout tuning.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    best = None

    for value in candidate_values:
        model_config = dict(base_model_config)
        candidate_learning_rate = learning_rate

        # LR tuning changes the optimizer; other searches modify the model configuration.
        if parameter_name == "learning_rate":
            candidate_learning_rate = value
        else:
            model_config[parameter_name] = value

        # Multi-head attention requires an integer number of features per head.
        if not valid_model_config(model_config):
            print(f"Skipping invalid config: {model_config}")
            continue

        display_name = "lr" if parameter_name == "learning_rate" else parameter_name

        print("\n" + "=" * 70)
        print(f"Training candidate {display_name} = {value}")
        print(f"Candidate config: {model_config}")
        print(f"Learning rate: {candidate_learning_rate}")
        print("=" * 70)

        _, history, model_state = train(
            model_cls=model_cls,
            model_config=model_config,
            tokenizer=tokenizer,
            make_dataloaders_fn=make_dataloaders_fn,
            learning_rate=candidate_learning_rate,
            device=device,
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            n_epochs=n_epochs,
            patience=patience,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
        )

        history_plot = output_dir / f"history_{display_name}_{checkpoint_tag(value)}.png"
        if experiment_name is None:
            history_title = f"{display_name}={value}"
        else:
            history_title = f"{experiment_name} {display_name}={value}"
        plot_history(history, history_plot, history_title)

        result = {
            parameter_name: float(value) if parameter_name in ["learning_rate", "dropout"] else value,
            "best_epoch": int(history["best_epoch"]),
            "best_dev_ppl": float(history["best_ppl"]),
            "history_plot": str(history_plot),
        }

        if parameter_name != "learning_rate":
            result["model_config"] = dict(model_config)

        results.append(result)

        # Compare candidates only on development perplexity; test is used after selection.
        if best is None or history["best_ppl"] < best["history"]["best_ppl"]:
            best = {
                "value": value,
                "learning_rate": float(candidate_learning_rate),
                "model_config": dict(model_config),
                "model_state": model_state,
                "history": history,
            }

    if best is None:
        raise ValueError(f"No valid configurations were trained for {parameter_name}.")

    return results, best


# Execute one complete search stage: tune, test the winner, plot, and save a report.
def run(
    parameter_name,
    candidate_values,
    model_cls,
    model_config,
    tokenizer,
    make_dataloaders_fn,
    learning_rate,
    device,
    output_dir,
    experiment_name,
    seed=42,
    train_batch_size=8,
    eval_batch_size=16,
    n_epochs=100,
    patience=3,
    weight_decay=0.01,
    grad_clip=None,
):
    """
    Run a one-parameter experiment, evaluate the winning candidate on test,
    save its checkpoint and write the report.

    It is used by baseline LR tuning, final LR readjustment, dropout tuning,
    and weight tying LR tuning.
    """
    results, best = tune(
        parameter_name=parameter_name,
        candidate_values=candidate_values,
        model_cls=model_cls,
        base_model_config=model_config,
        tokenizer=tokenizer,
        make_dataloaders_fn=make_dataloaders_fn,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_dir,
        seed=seed,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        n_epochs=n_epochs,
        patience=patience,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        experiment_name=experiment_name if parameter_name == "learning_rate" else None,
    )

    # Store metadata using the field names expected by the following experiment stages.
    if parameter_name == "learning_rate":
        checkpoint_extra = {
            "best_learning_rate": float(best["learning_rate"]),
            "best_dev_ppl": float(best["history"]["best_ppl"]),
        }
        plot_filename = "learning_rate_tuning.png"
        plot_title = "Learning-rate tuning"
        plot_key = "lr_plot"
    else:
        checkpoint_extra = {
            "learning_rate": float(best["learning_rate"]),
            "dev_ppl": float(best["history"]["best_ppl"]),
        }
        plot_filename = f"{parameter_name}_tuning.png"
        plot_title = f"{parameter_name.capitalize()} tuning"
        plot_key = f"{parameter_name}_plot"

    checkpoint_path, test_ppl, test_loss = save_best_model(
        output_dir=output_dir,
        model_cls=model_cls,
        model_config=best["model_config"],
        model_state=best["model_state"],
        tokenizer=tokenizer,
        make_dataloaders_fn=make_dataloaders_fn,
        device=device,
        seed=seed,
        eval_batch_size=eval_batch_size,
        checkpoint_extra=checkpoint_extra,
    )

    plot_path = Path(output_dir) / plot_filename
    plot_tuning_results(results, plot_path, parameter_name, plot_title)

    report = {
        "experiment_name": experiment_name,
        "seed": seed,
        "device": str(device),
        "model_class": model_cls.__name__,
        "fixed_training_hyperparameters": fixed_training_hyperparameters(
            train_batch_size,
            eval_batch_size,
            n_epochs,
            patience,
            weight_decay,
            grad_clip,
        ),
        "results": results,
        "best": {
            "learning_rate": float(best["learning_rate"]),
            "best_epoch": int(best["history"]["best_epoch"]),
            "dev_ppl": float(best["history"]["best_ppl"]),
            "test_loss": float(test_loss),
            "test_ppl": float(test_ppl),
            "model_config": dict(best["model_config"]),
            "model_path": str(checkpoint_path),
            plot_key: str(plot_path),
        },
    }

    if parameter_name == "learning_rate":
        report["tested_learning_rates"] = [float(value) for value in candidate_values]
        report["fixed_model_hyperparameters"] = dict(model_config)
    elif parameter_name == "dropout":
        report["learning_rate"] = float(learning_rate)
        report["base_model_config"] = dict(model_config)
        report["tested_dropout_values"] = [float(value) for value in candidate_values]

    save_report(report, Path(output_dir) / "report.json")
    return report


# Evaluate the selected model on test and serialize everything needed to reload it.
def save_best_model(
    output_dir,
    model_cls,
    model_config,
    model_state,
    tokenizer,
    make_dataloaders_fn,
    device,
    seed,
    eval_batch_size,
    checkpoint_extra,
):
    _, test_ppl, test_loss = eval_model_saved(
        model_cls=model_cls,
        model_config=model_config,
        model_state=model_state,
        tokenizer=tokenizer,
        make_dataloaders_fn=make_dataloaders_fn,
        device=device,
        eval_batch_size=eval_batch_size,
        seed=seed,
    )

    # Every stage writes one consistently named best checkpoint inside its own output folder.
    checkpoint_path = Path(output_dir) / "best_model.pt"

    torch.save(
        {
            "model_state_dict": model_state,
            "model_config": dict(model_config),
            "model_class": model_cls.__name__,
            **checkpoint_extra,
            "test_loss": float(test_loss),
            "test_ppl": float(test_ppl),
        },
        checkpoint_path,
    )

    return checkpoint_path, float(test_ppl), float(test_loss)


# Save the train/dev loss curves for one candidate.
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


# Save validation perplexity against the candidate values tested in one stage.
def plot_tuning_results(results, output_path, parameter_name, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [item[parameter_name] for item in results]
    dev_ppls = [item["best_dev_ppl"] for item in results]

    plt.figure()
    plt.plot(x_values, dev_ppls, marker="o")

    if parameter_name == "learning_rate":
        plt.xscale("log")
        plt.xlabel("learning rate")
    else:
        plt.xlabel(parameter_name)

    plt.ylabel("best dev perplexity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def checkpoint_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


# Write both a machine-readable JSON report and a compact text summary.
def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"{report['experiment_name']} report\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"seed: {report['seed']}\n")
        f.write(f"device: {report['device']}\n")
        f.write(f"model_class: {report.get('model_class')}\n")
        f.write(f"learning_rate: {report.get('learning_rate', report.get('best', {}).get('learning_rate'))}\n")
        f.write(f"fixed_model_hyperparameters: {report.get('fixed_model_hyperparameters')}\n")
        f.write(f"fixed_training_hyperparameters: {report.get('fixed_training_hyperparameters')}\n\n")

        if "results" in report:
            f.write("Results:\n")
            for item in report["results"]:
                f.write(json.dumps(item, indent=4))
                f.write("\n")

        if "steps" in report:
            f.write("\nGreedy steps:\n")
            for step in report["steps"]:
                f.write(json.dumps(step.get("best", step), indent=4))
                f.write("\n")

        f.write("\nBest configuration:\n")
        f.write(json.dumps(report["best"], indent=4))
        f.write("\n")


# Missing reports return None so a caller can use defaults or raise a clear error.
def load_report(path):
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# The embedding size must split evenly across the requested attention heads.
def valid_model_config(config):
    return config["d_model"] % config["n_heads"] == 0


def fixed_training_hyperparameters(
    train_batch_size,
    eval_batch_size,
    n_epochs,
    patience,
    weight_decay,
    grad_clip,
):
    return {
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "n_epochs": n_epochs,
        "patience": patience,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "optimizer": "AdamW",
    }
