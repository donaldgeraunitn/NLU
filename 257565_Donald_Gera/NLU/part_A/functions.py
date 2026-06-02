import json
import math
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from utils import PAD_TOKEN

try:
    from conll import evaluate as conll_evaluate
except ImportError:
    conll_evaluate = None


# Seed Python, NumPy and PyTorch to make experiments reproducible.
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


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# Train for one epoch using the joint intent-classification and slot-filling loss.
def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []

    for batch in data:
        optimizer.zero_grad()  # Zeroing the gradient

        slots, intent = model(batch["utterances"], batch["slots_len"])
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss

        loss_intent = criterion_intents(intent, batch["intents"])
        loss_slot = criterion_slots(slots, batch["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.

        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        optimizer.step()  # Update the weights

    return loss_array


def normalize_slot_label(label):
    # PAD and CLS must not be interpreted as real slot classes by CoNLL evaluation.
    if label in {"pad", "cls"}:
        return "O"
    return label


# Evaluate both tasks and convert predictions back to labels for CoNLL slot scoring.
def eval_loop(data, criterion_slots, criterion_intents, model, lang, show_progress = True):
    if conll_evaluate is None:
        raise ImportError("conll.py was not found. Copy conll.py into the project folder.")

    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): 
        batches = tqdm(data, desc="Evaluating:", unit="batch") if show_progress else data
        for batch in batches:
            slots, intents = model(batch["utterances"], batch["slots_len"])
            slots = slots.permute(0, 2, 1)  # We need this for computing the loss

            loss_intent = criterion_intents(intents, batch["intents"])
            loss_slot = criterion_slots(slots, batch["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference: use the class with the highest sentence-level score.
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in batch["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference: ignore the final synthetic CLS position and padding.
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = batch["slots_len"].tolist()[id_seq] - 1  # Ignore CLS

                utt_ids = batch["utterances"][id_seq][:length].tolist()
                gt_ids = batch["y_slots"][id_seq][:length].tolist()
                pred_ids = seq[:length].tolist()

                utterance = [lang.id2word[elem] for elem in utt_ids]
                gt_slots = [normalize_slot_label(lang.id2slot[elem]) for elem in gt_ids]
                pred_slots = [normalize_slot_label(lang.id2slot[elem]) for elem in pred_ids]

                ref_slots.append(list(zip(utterance, gt_slots)))
                hyp_slots.append(list(zip(utterance, pred_slots)))

    results_slots = conll_evaluate(ref_slots, hyp_slots)
    results_intents = classification_report(
        ref_intents,
        hyp_intents,
        zero_division=0,
        output_dict=True,
    )

    return results_slots, results_intents, loss_array


def get_slot_f1(results):
    return float(results.get("total", {}).get("f", 0.0))


def get_intent_accuracy(intent_report):
    return float(intent_report.get("accuracy", 0.0))


# The development metric used for model selection is configurable from the CLI.
def score_for_selection(slot_f1, intent_acc, selection_metric="slot_f1"):
    if selection_metric == "average":
        return (slot_f1 + intent_acc) / 2.0
    if selection_metric == "intent_acc":
        return intent_acc
    return slot_f1


def model_state_on_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def checkpoint_tag(value):
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_")


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    with open(output_path.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"{report['experiment_name']} report\n")
        f.write("=" * 40 + "\n\n")
        f.write(json.dumps(report, indent=4))
        f.write("\n")


def load_report(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def plot_tuning_results(results, parameter_name, output_path):
    if parameter_name is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values = [result["value"] for result in results]
    means = [result["dev"]["selection_score"]["mean"] for result in results]
    stds = [result["dev"]["selection_score"]["std"] for result in results]

    plt.figure()
    plt.errorbar(x_values, means, yerr=stds, marker="o", capsize=4)
    if parameter_name == "learning_rate":
        plt.xscale("log")
    plt.xlabel(parameter_name)
    plt.ylabel("dev selection score")
    plt.title(f"Tuning {parameter_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def valid_model_config(model_config):
    return model_config["d_model"] % model_config["n_heads"] == 0


# Rebuild a model from a state dictionary and evaluate one dataset split.
def evaluate_state(
    model_cls,
    model_config,
    model_state,
    data_loader,
    lang,
    device,
):
    model = model_cls(
        vocab_size=len(lang.word2id),
        slots_size=len(lang.id2slot),
        n_intents=len(lang.intent2id),
        **model_config,
    ).to(device)
    model.load_state_dict(model_state)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    results_slots, results_intents, losses = eval_loop(
        data_loader,
        criterion_slots,
        criterion_intents,
        model,
        lang,
    )

    return {
        "loss": float(np.asarray(losses).mean()),
        "slot_f1": get_slot_f1(results_slots),
        "intent_acc": get_intent_accuracy(results_intents),
    }


def eval_model_saved(
    model_cls,
    make_dataloaders_fn,
    checkpoint_path,
    device,
    dataset_dir="dataset/ATIS",
    eval_batch_size=64,
):
    """
    Load a saved checkpoint and evaluate it on the test set without training.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # The checkpoint stores both the weights and the architecture needed to rebuild the model.
    checkpoint = torch.load(checkpoint_path, map_location=device)

    required_keys = {"model_state_dict", "model_config"}
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    seed = int(checkpoint.get("seed", 42))
    data_seed = int(checkpoint.get("data_seed", 42))
    set_seed(seed)

    _, _, test_loader, lang, _, _, _ = make_dataloaders_fn(
        device=device,
        dataset_dir=dataset_dir,
        train_batch_size=1,
        eval_batch_size=eval_batch_size,
        seed=seed,
        data_seed=data_seed,
    )

    # Evaluation is valid only if token, slot and intent mappings match training.
    if "lang" in checkpoint and checkpoint["lang"] != lang.to_dict():
        raise ValueError(
            "The language mappings rebuilt from the dataset do not match the "
            "mappings stored in the checkpoint. Check the dataset and data seed."
        )

    metrics = evaluate_state(
        model_cls=model_cls,
        model_config=checkpoint["model_config"],
        model_state=checkpoint["model_state_dict"],
        data_loader=test_loader,
        lang=lang,
        device=device,
    )

    print("\nTest results")
    print("-" * 40)
    print(f"Test loss:       {metrics['loss']:.4f}")
    print(f"Test slot F1:    {metrics['slot_f1']:.4f}")
    print(f"Test intent acc: {metrics['intent_acc']:.4f}")
    print(f"Model config:    {checkpoint['model_config']}")
    print(f"Checkpoint:      {checkpoint_path}")

    return metrics


def train(
    model_cls,
    model_config,
    make_dataloaders_fn,
    learning_rate,
    device,
    output_dir,
    seed=42,
    data_seed=42,
    dataset_dir="dataset/ATIS",
    train_batch_size=128,
    eval_batch_size=64,
    n_epochs=200,
    patience=3,
    eval_every=5,
    weight_decay=0.01,
    selection_metric="slot_f1",
):
    """
    Train one model from scratch.

    This is the only training function. It creates the model, optimizer and losses,
    performs epoch training, validates periodically, applies early stopping,
    restores the best weights, saves the checkpoint and returns the best dev metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # Build the fixed split, vocabulary and dataloaders before creating the model.
    train_loader, dev_loader, test_loader, lang, _, _, _ = make_dataloaders_fn(
        device=device,
        dataset_dir=dataset_dir,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        data_seed=data_seed,
    )

    model = model_cls(
        vocab_size=len(lang.word2id),
        slots_size=len(lang.id2slot),
        n_intents=len(lang.intent2id),
        **model_config,
    ).to(device)
    model.apply(init_weights)

    # Slot padding positions are ignored, while each utterance contributes one intent label.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    sampled_epochs = []
    losses_train = []
    losses_dev = []

    best_score = -math.inf
    best_slot_f1 = -1.0
    best_intent_acc = 0.0
    best_epoch = -1
    best_model_state = None
    remaining_patience = patience

    pbar = tqdm(range(1, n_epochs + 1), desc="Epochs", dynamic_ncols=True, leave=True)

    for epoch in pbar:
        train_losses = train_loop(
            train_loader,
            optimizer,
            criterion_slots,
            criterion_intents,
            model,
        )
        train_loss = float(np.asarray(train_losses).mean())

        # Validation is intentionally sampled every `eval_every` epochs.
        should_validate = epoch % eval_every == 0 or epoch == n_epochs
        if not should_validate:
            pbar.set_postfix(
                epoch=epoch,
                train_loss=f"{train_loss:.4f}",
                best_f1=f"{best_slot_f1:.2f}",
                patience=remaining_patience,
                refresh=True,
            )
            continue

        results_dev, intent_dev, dev_losses = eval_loop(
            dev_loader,
            criterion_slots,
            criterion_intents,
            model,
            lang,
        )

        dev_loss = float(np.asarray(dev_losses).mean())
        slot_f1 = get_slot_f1(results_dev)
        intent_acc = get_intent_accuracy(intent_dev)
        current_score = score_for_selection(slot_f1, intent_acc, selection_metric)

        sampled_epochs.append(epoch)
        losses_train.append(train_loss)
        losses_dev.append(dev_loss)

        # Early stopping follows the selected development metric, never the test set.
        if best_model_state is None or current_score > best_score:
            best_score = current_score
            best_slot_f1 = slot_f1
            best_intent_acc = intent_acc
            best_epoch = epoch
            best_model_state = model_state_on_cpu(model)
            remaining_patience = patience
        else:
            remaining_patience -= 1

        pbar.set_postfix(
            epoch=epoch,
            train_loss=f"{train_loss:.4f}",
            dev_loss=f"{dev_loss:.4f}",
            slot_f1=f"{slot_f1:.2f}",
            intent_acc=f"{intent_acc:.4f}",
            best_f1=f"{best_slot_f1:.2f}",
            patience=remaining_patience,
            refresh=True,
        )

        if remaining_patience <= 0:
            pbar.set_postfix(
                epoch=epoch,
                train_loss=f"{train_loss:.4f}",
                dev_loss=f"{dev_loss:.4f}",
                slot_f1=f"{slot_f1:.2f}",
                intent_acc=f"{intent_acc:.4f}",
                best_f1=f"{best_slot_f1:.2f}",
                patience=0,
                status="early_stop",
                refresh=True,
            )
            break

    if best_model_state is None:
        raise RuntimeError("Training ended before the first validation step.")

    model.load_state_dict(best_model_state)

    history = {
        "sampled_epochs": sampled_epochs,
        "losses_train": losses_train,
        "losses_dev": losses_dev,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_slot_f1": float(best_slot_f1),
        "best_intent_acc": float(best_intent_acc),
        "selection_metric": selection_metric,
    }

    history_plot = output_dir / "history.png"
    plot_history(history, history_plot, title=f"seed={seed}")

    # Each run keeps its own checkpoint; tuning later copies the best one to best_model.pt.
    checkpoint_path = output_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": best_model_state,
            "model_config": dict(model_config),
            "model_class": model_cls.__name__,
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "data_seed": int(data_seed),
            "best_epoch": int(best_epoch),
            "dev_slot_f1": float(best_slot_f1),
            "dev_intent_acc": float(best_intent_acc),
            "selection_score": float(best_score),
            "history": history,
            "lang": lang.to_dict(),
        },
        checkpoint_path,
    )

    return {
        "seed": int(seed),
        "data_seed": int(data_seed),
        "best_epoch": int(best_epoch),
        "dev_slot_f1": float(best_slot_f1),
        "dev_intent_acc": float(best_intent_acc),
        "selection_score": float(best_score),
        "checkpoint_path": str(checkpoint_path),
        "history_plot": str(history_plot),
        "model_state": best_model_state,
        "lang": lang,
        "test_loader": test_loader,
    }


def tune(
    parameter_name,
    candidate_values,
    model_cls,
    base_model_config,
    base_learning_rate,
    make_dataloaders_fn,
    device,
    output_dir,
    experiment_name,
    runs_per_config=1,
    seed=42,
    data_seed=42,
    dataset_dir="dataset/ATIS",
    train_batch_size=128,
    eval_batch_size=64,
    n_epochs=200,
    patience=3,
    eval_every=5,
    weight_decay=0.01,
    selection_metric="slot_f1",
    evaluate_test=False,
):
    """
    Generic tuning function used for every experiment.

    parameter_name can be:
      - "learning_rate"
      - one model hyperparameter, such as "d_model" or "dropout"
      - None, to repeat a fixed final/custom configuration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if parameter_name is None:
        candidate_values = [None]

    candidate_results = []
    best_candidate_internal = None

    # Train every candidate independently; repeated runs use consecutive model seeds.
    for value in candidate_values:
        model_config = dict(base_model_config)
        learning_rate = base_learning_rate

        if parameter_name == "learning_rate":
            learning_rate = float(value)
        elif parameter_name is not None:
            model_config[parameter_name] = value

        if learning_rate is None:
            raise ValueError("A learning rate must be provided.")

        if not valid_model_config(model_config):
            print(f"Skipping invalid config: {model_config}")
            continue

        value_tag = "fixed" if parameter_name is None else f"{parameter_name}_{checkpoint_tag(value)}"
        run_results = []

        print("\n" + "=" * 70)
        if parameter_name is None:
            print(f"Experiment: {experiment_name}")
        else:
            print(f"Tuning {parameter_name}: {value}")
        print(f"Model config: {model_config}")
        print(f"Learning rate: {learning_rate}")
        print("=" * 70)

        for run_idx in range(1, runs_per_config + 1):
            run_seed = seed + run_idx - 1
            run_dir = output_dir / value_tag / f"run_{run_idx:02d}_seed_{run_seed}"

            print(f"\nRun {run_idx}/{runs_per_config} | seed={run_seed} | data_seed={data_seed}")

            run = train(
                model_cls=model_cls,
                model_config=model_config,
                make_dataloaders_fn=make_dataloaders_fn,
                learning_rate=learning_rate,
                device=device,
                output_dir=run_dir,
                seed=run_seed,
                data_seed=data_seed,
                dataset_dir=dataset_dir,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                n_epochs=n_epochs,
                patience=patience,
                eval_every=eval_every,
                weight_decay=weight_decay,
                selection_metric=selection_metric,
            )
            run_results.append(run)

        # Aggregate repeated runs before comparing candidate configurations.
        dev_summary = {
            "slot_f1": mean_std([run["dev_slot_f1"] for run in run_results]),
            "intent_acc": mean_std([run["dev_intent_acc"] for run in run_results]),
            "selection_score": mean_std([run["selection_score"] for run in run_results]),
        }

        candidate_public = {
            "value": value,
            "model_config": dict(model_config),
            "learning_rate": float(learning_rate),
            "dev": dev_summary,
            "runs": [
                {
                    "seed": run["seed"],
                    "data_seed": run["data_seed"],
                    "best_epoch": run["best_epoch"],
                    "dev_slot_f1": run["dev_slot_f1"],
                    "dev_intent_acc": run["dev_intent_acc"],
                    "selection_score": run["selection_score"],
                    "checkpoint_path": run["checkpoint_path"],
                    "history_plot": run["history_plot"],
                }
                for run in run_results
            ],
        }
        candidate_results.append(candidate_public)

        candidate_internal = {
            "public": candidate_public,
            "runs": run_results,
        }

        if (
            best_candidate_internal is None
            or dev_summary["selection_score"]["mean"]
            > best_candidate_internal["public"]["dev"]["selection_score"]["mean"]
        ):
            best_candidate_internal = candidate_internal

    if best_candidate_internal is None:
        raise ValueError("No valid candidate configurations were trained.")

    best_public = best_candidate_internal["public"]
    best_runs = best_candidate_internal["runs"]

    # Keep a convenient copy of the best individual checkpoint from the winning candidate.
    # This is the file that should be transferred to the submission `bin/` folder.
    best_run = max(best_runs, key=lambda run: run["selection_score"])
    best_model_path = output_dir / "best_model.pt"
    shutil.copy2(best_run["checkpoint_path"], best_model_path)

    best = {
        "parameter_name": parameter_name,
        "value": best_public["value"],
        "model_config": dict(best_public["model_config"]),
        "learning_rate": float(best_public["learning_rate"]),
        "dev": best_public["dev"],
        "best_individual_checkpoint": str(best_model_path),
    }

    # Test is evaluated only after the winning candidate has been selected on dev.
    if evaluate_test:
        for public_run, internal_run in zip(best_public["runs"], best_runs):
            test_metrics = evaluate_state(
                model_cls=model_cls,
                model_config=best_public["model_config"],
                model_state=internal_run["model_state"],
                data_loader=internal_run["test_loader"],
                lang=internal_run["lang"],
                device=device,
            )
            public_run["test_loss"] = test_metrics["loss"]
            public_run["test_slot_f1"] = test_metrics["slot_f1"]
            public_run["test_intent_acc"] = test_metrics["intent_acc"]

        best["test"] = {
            "loss": mean_std([run["test_loss"] for run in best_public["runs"]]),
            "slot_f1": mean_std([run["test_slot_f1"] for run in best_public["runs"]]),
            "intent_acc": mean_std([run["test_intent_acc"] for run in best_public["runs"]]),
        }

    plot_path = plot_tuning_results(
        candidate_results,
        parameter_name,
        output_dir / "tuning.png",
    )

    report = {
        "experiment_name": experiment_name,
        "parameter_name": parameter_name,
        "candidate_values": list(candidate_values),
        "runs_per_config": int(runs_per_config),
        "seed": int(seed),
        "data_seed": int(data_seed),
        "dataset_dir": dataset_dir,
        "selection_metric": selection_metric,
        "fixed_training_hyperparameters": {
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "n_epochs": n_epochs,
            "patience": patience,
            "eval_every": eval_every,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
        },
        "base_model_config": dict(base_model_config),
        "base_learning_rate": base_learning_rate,
        "results": candidate_results,
        "best": best,
    }

    if plot_path is not None:
        report["tuning_plot"] = str(plot_path)

    save_report(report, output_dir / "report.json")
    return report
