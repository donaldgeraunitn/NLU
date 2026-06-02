## Install the dependencies

Open a terminal in the root directory of the extracted archive and run:

```bash
pip install -r requirements.txt
```

## Run all submitted model evaluations

From the root directory, run:

```bash
python evaluate_all_models.py
```

The launcher evaluates the submitted checkpoints sequentially without retraining the models:

1. `LM/part_A/bin/best_model.pt`
2. `LM/part_B/bin/best_model.pt`
3. `NLU/part_A/bin/best_model.pt`
4. `NLU/part_B/bin/best_model.pt`

The LM scripts report test loss and perplexity. The NLU scripts report test loss, slot F1, and intent accuracy.

## Run one evaluation manually

The root launcher is the recommended entry point. The following commands are equivalent when an individual part needs to be evaluated separately.

### LM - Part A

```bash
cd LM/part_A
python main.py --eval --checkpoint bin/best_model.pt
cd ../..
```

### LM - Part B

```bash
cd LM/part_B
python main.py --eval --checkpoint bin/best_model.pt
cd ../..
```

### NLU - Part A

```bash
cd NLU/part_A
python main.py --eval --checkpoint bin/best_model.pt
cd ../..
```

### NLU - Part B: BERT

```bash
cd NLU/part_B
python main.py --eval --model bert --checkpoint bin/bert/best_model.pt
cd ../..
```

### NLU - Part B: GPT-2

```bash
cd NLU/part_B
python main.py --eval --model gpt2 --checkpoint bin/gpt2/best_model.pt
cd ../..
```

Run the manual commands from the indicated part directory because dataset paths are relative to each project part.

## Notes

- `conll.py` must remain in both NLU part directories because it is used to compute the slot-filling F1 score.
- The files under `bin/` are the trained checkpoints submitted for evaluation.
- Training instructions and part-specific CLI options are documented in the `README.md` file inside each project part.
