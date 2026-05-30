import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm.auto import tqdm

from model import make_model
from utils import IGNORE_INDEX

try:
    from conll import evaluate
except ImportError:
    evaluate = None


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    model.train()
    loss_array = []

    for batch in data:
        optimizer.zero_grad()  # Zeroing the gradient

        slots, intent = model(batch["utterances"], batch["attention_mask"])
        slots = slots.permute(0, 2, 1)  # We need this for computing the loss

        loss_intent = criterion_intents(intent, batch["intents"])
        loss_slot = criterion_slots(slots, batch["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        optimizer.step()  # Update the weights

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    if evaluate is None:
        raise ImportError(
            "conll.py was not found. Place the CoNLL evaluation script in the project folder."
        )

    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():  # It avoids the creation of the computational graph
        for batch in data:
            slots, intents = model(batch["utterances"], batch["attention_mask"])
            slots = slots.permute(0, 2, 1)  # We need this for computing the loss

            loss_intent = criterion_intents(intents, batch["intents"])
            loss_slot = criterion_slots(slots, batch["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [
                lang.id2intent[idx]
                for idx in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[idx] for idx in batch["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference: decode the sub-token selected during collation.
            output_slots = torch.argmax(slots, dim=1)
            for idx, seq in enumerate(output_slots):
                active_positions = batch["y_slots"][idx] != IGNORE_INDEX
                gt_ids = batch["y_slots"][idx][active_positions].tolist()
                hyp_ids = seq[active_positions].tolist()
                words = batch["words"][idx]

                if len(words) != len(gt_ids):
                    raise ValueError("Words and aligned slot labels have different lengths.")

                gt_slots = [lang.id2slot[slot_id] for slot_id in gt_ids]
                out_slots = [lang.id2slot[slot_id] for slot_id in hyp_ids]

                ref_slots.append(list(zip(words, gt_slots)))
                hyp_slots.append(list(zip(words, out_slots)))

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF.
        print("Warning:", ex)
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents,
        hyp_intents,
        zero_division=0,
        output_dict=True,
    )

    return results, report_intent, loss_array


def save_model(
    path,
    model,
    optimizer,
    epoch,
    model_type,
    model_name,
    lang,
    dev_slot_f1,
    dev_intent_accuracy,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_type": model_type,
            "model_name": model_name,
            "slot2id": lang.slot2id,
            "intent2id": lang.intent2id,
            "dev_slot_f1": dev_slot_f1,
            "dev_intent_accuracy": dev_intent_accuracy,
        },
        path,
    )


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_model(
    train_loader,
    dev_loader,
    test_loader,
    lang,
    tokenizer,
    device,
    model_type,
    model_name,
    output_dir,
    run_id=0,
    seed=42,
    lr=5e-5,
    dropout=0.1,
    weight_decay=0.0,
    optimizer_name="adamw",
    n_epochs=20,
    patience=3,
    eval_every=1,
):
    set_seed(seed)
    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)

    model = make_model(
        model_type=model_type,
        model_name=model_name,
        slots_size=len(lang.slot2id),
        n_intents=len(lang.intent2id),
        tokenizer=tokenizer,
        dropout=dropout,
    ).to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_intents = nn.CrossEntropyLoss()

    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"{model_type}_run_{run_id}.pt"

    best_f1 = -1
    current_patience = patience
    pbar = tqdm(range(n_epochs), desc=f"{model_type.upper()} run {run_id + 1}")

    for epoch in pbar:
        train_loss = train_loop(
            train_loader,
            optimizer,
            criterion_slots,
            criterion_intents,
            model,
        )

        if epoch % eval_every != 0:
            continue

        results_dev, intent_dev, loss_dev = eval_loop(
            dev_loader,
            criterion_slots,
            criterion_intents,
            model,
            lang,
        )

        slot_f1 = results_dev["total"]["f"]
        intent_accuracy = intent_dev["accuracy"]
        pbar.set_description(
            f"{model_type.upper()} run {run_id + 1} | "
            f"Slot F1: {slot_f1:.2f}; Intent Acc: {intent_accuracy:.4f}; "
            f"Train Loss: {np.mean(train_loss):.4f}; Dev Loss: {np.mean(loss_dev):.4f}"
        )

        if slot_f1 > best_f1:
            best_f1 = slot_f1
            current_patience = patience
            save_model(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                model_type,
                model_name,
                lang,
                slot_f1,
                intent_accuracy,
            )
        else:
            current_patience -= 1

        if current_patience <= 0:  # Early stopping with patience
            break

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])

    results_test, intent_test, _ = eval_loop(
        test_loader,
        criterion_slots,
        criterion_intents,
        model,
        lang,
    )

    return {
        "run": run_id,
        "seed": seed,
        "best_epoch": checkpoint["epoch"],
        "dev_slot_f1": checkpoint["dev_slot_f1"],
        "dev_intent_accuracy": checkpoint["dev_intent_accuracy"],
        "test_slot_f1": results_test["total"]["f"],
        "test_intent_accuracy": intent_test["accuracy"],
        "checkpoint": str(checkpoint_path),
    }


def run_experiment(
    train_loader,
    dev_loader,
    test_loader,
    lang,
    tokenizer,
    device,
    model_type,
    model_name,
    output_dir,
    runs=5,
    seed=42,
    lr=5e-5,
    dropout=0.1,
    weight_decay=0.0,
    optimizer_name="adamw",
    n_epochs=20,
    patience=3,
    eval_every=1,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_results = []
    for run_id in range(runs):
        run_results.append(
            train_model(
                train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                lang=lang,
                tokenizer=tokenizer,
                device=device,
                model_type=model_type,
                model_name=model_name,
                output_dir=output_dir,
                run_id=run_id,
                seed=seed + run_id,
                lr=lr,
                dropout=dropout,
                weight_decay=weight_decay,
                optimizer_name=optimizer_name,
                n_epochs=n_epochs,
                patience=patience,
                eval_every=eval_every,
            )
        )

    slot_f1s = np.asarray([result["test_slot_f1"] for result in run_results])
    intent_accs = np.asarray(
        [result["test_intent_accuracy"] for result in run_results]
    )

    summary = {
        "model_type": model_type,
        "model_name": model_name,
        "optimizer": optimizer_name,
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "runs": run_results,
        "test_slot_f1_mean": float(slot_f1s.mean()),
        "test_slot_f1_std": float(slot_f1s.std()),
        "test_intent_accuracy_mean": float(intent_accs.mean()),
        "test_intent_accuracy_std": float(intent_accs.std()),
    }

    with open(output_dir / f"{model_type}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
    print(
        "Intent Acc",
        round(intent_accs.mean(), 3),
        "+-",
        round(intent_accs.std(), 3),
    )

    return summary


def eval_model_saved(checkpoint_path, test_loader, lang, tokenizer, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = make_model(
        model_type=checkpoint["model_type"],
        model_name=checkpoint["model_name"],
        slots_size=len(checkpoint["slot2id"]),
        n_intents=len(checkpoint["intent2id"]),
        tokenizer=tokenizer,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_intents = nn.CrossEntropyLoss()

    results_test, intent_test, _ = eval_loop(
        test_loader,
        criterion_slots,
        criterion_intents,
        model,
        lang,
    )

    print("Slot F1:", results_test["total"]["f"])
    print("Intent Accuracy:", intent_test["accuracy"])

    return results_test, intent_test
