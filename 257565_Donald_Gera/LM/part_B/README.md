## Dependencies

Install the dependencies from the global submission-level `requirements.txt` file before running the script.

The pretrained GPT-2 backbone is loaded with `from_pretrained`. Therefore, it must either be available in the local Hugging Face cache or downloadable when the script is executed.

## Evaluate the submitted best model

The submitted checkpoint is expected at:

```text
bin/best_model.pt
```

Evaluate it on the Penn Treebank test split with:

```bash
python main.py --eval --checkpoint bin/best_model.pt
```

The script reads the pretrained model name, LoRA rank, and LoRA alpha value stored in the checkpoint, rebuilds the corresponding model, restores the saved state, and prints the test loss and test perplexity.

## Train one configuration

Running the script without `--experiments` trains one configuration. The default values are `rank=4` and `alpha=32`:

```bash
python main.py
```

A specific configuration can be selected with:

```bash
python main.py --rank 4 --alpha 32
```

The best checkpoint for the execution is written to:

```text
outputs/best_model.pt
```

## Run all predefined experiments

Use `--experiments` to train every combination of the predefined rank and alpha values:

```bash
python main.py --experiments
```

The default search space is:

```text
rank:  2, 4, 8, 16
alpha: 8, 16, 32
```

Custom search values can be supplied with:

```bash
python main.py --experiments --rank_values 2 4 8 16 --alpha_values 8 16 32
```

For each configuration, the script stores a loss plot and evaluates test perplexity. The checkpoint with the lowest development perplexity is retained as `outputs/best_model.pt`. A JSON report, a text summary, and a tuning plot are also written under `outputs/`.

## Optional comparison with Part 1.A

To record whether the selected LoRA model improves over the Part 1.A test perplexity, pass the Part 1.A value explicitly:

```bash
python main.py --experiments --part_a_ppl <PART_A_TEST_PPL>
```

## Main CLI arguments

| Argument | Purpose | Default |
|---|---|---|
| `--experiments` | Train all requested rank-alpha combinations instead of one configuration | disabled |
| `--rank` | LoRA rank for a single run | `4` |
| `--alpha` | LoRA alpha for a single run | `32` |
| `--rank_values` | Rank values used with `--experiments` | `2 4 8 16` |
| `--alpha_values` | Alpha values used with `--experiments` | `8 16 32` |
| `--model_name` | Pretrained GPT-2 checkpoint | `openai-community/gpt2` |
| `--output_dir` | Directory for generated checkpoints, plots, and reports | `outputs` |
| `--learning_rate` | AdamW learning rate | `2e-4` |
| `--train_batch_size` | Training batch size | `8` |
| `--eval_batch_size` | Development and test batch size | `16` |
| `--n_epochs` | Maximum number of epochs | `8` |
| `--patience` | Early-stopping patience | `3` |
| `--weight_decay` | AdamW weight decay | `0.01` |
| `--seed` | Random seed | `42` |
| `--device` | Explicit PyTorch device, such as `cpu` or `cuda:0` | automatic |
| `--eval` | Skip training and evaluate a saved checkpoint | disabled |
| `--checkpoint` | Saved checkpoint path used with `--eval` | none |
| `--part_a_ppl` | Optional Part 1.A test perplexity for comparison | none |
