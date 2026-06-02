# LM and NLU Exam Submission

This archive contains the final implementations, datasets, reports, and trained checkpoints for the LM and NLU exam exercises.

The submitted checkpoints are stored under the corresponding `bin/` directories. The `outputs/` folders are used only for training artifacts and are not required to run the final evaluations.

## Project structure

```text
257565_Donald_Gera/
├── README.md
├── requirements.txt
├── evaluate_all_models.py
├── LM/
│   ├── report.pdf
│   ├── part_A/
│   │   ├── main.py
│   │   ├── functions.py
│   │   ├── utils.py
│   │   ├── model.py
│   │   ├── README.md
│   │   ├── dataset/
│   │   └── bin/
│   │       └── best_model.pt
│   └── part_B/
│       ├── main.py
│       ├── functions.py
│       ├── utils.py
│       ├── model.py
│       ├── README.md
│       ├── dataset/
│       └── bin/
│           └── best_model.pt
└── NLU/
    ├── report.pdf
    ├── part_A/
    │   ├── main.py
    │   ├── functions.py
    │   ├── utils.py
    │   ├── model.py
    │   ├── conll.py
    │   ├── README.md
    │   ├── dataset/
    │   └── bin/
    │       └── best_model.pt
    └── part_B/
        ├── main.py
        ├── functions.py
        ├── utils.py
        ├── model.py
        ├── conll.py
        ├── README.md
        ├── dataset/
        └── bin/
            ├── bert/
            │   └── best_model.pt
            └── gpt2/
                └── best_model.pt
```

## Install the dependencies

Open a terminal in the root directory of the extracted archive and run:

```bash
pip install -r requirements.txt
```

A GPU is optional. The scripts use CUDA when it is available and fall back to CPU otherwise.

Some parts load Hugging Face tokenizers or pretrained checkpoints. On the first execution, an internet connection is required unless the required files are already available in the local Hugging Face cache.

## Run all submitted model evaluations

From the root directory, run:

```bash
python evaluate_all_models.py
```

The launcher evaluates the submitted checkpoints sequentially without retraining the models:

1. `LM/part_A/bin/best_model.pt`
2. `LM/part_B/bin/best_model.pt`
3. `NLU/part_A/bin/best_model.pt`
4. `NLU/part_B/bin/bert/best_model.pt`
5. `NLU/part_B/bin/gpt2/best_model.pt`

The LM scripts report test loss and perplexity. The NLU scripts report test loss, slot F1, and intent accuracy.

## Run one evaluation manually

The root launcher is the recommended entry point. The following commands are equivalent when an individual part needs to be evaluated separately.

### LM - Part 1.A

```bash
cd LM/part_A
python main.py --eval --checkpoint bin/best_model.pt
cd ../..
```

### LM - Part 1.B

```bash
cd LM/part_B
python main.py --eval --checkpoint bin/best_model.pt
cd ../..
```

### NLU - Part 2.A

```bash
cd NLU/part_A
python main.py --eval --checkpoint bin/best_model.pt
cd ../..
```

### NLU - Part 2.B: BERT

```bash
cd NLU/part_B
python main.py --eval --model bert --checkpoint bin/bert/best_model.pt
cd ../..
```

### NLU - Part 2.B: GPT-2

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
- Jupyter notebooks are not required to execute the submitted solution.
