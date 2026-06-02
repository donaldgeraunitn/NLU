## Dependencies

Install the dependencies from the submission-level `requirements.txt` file. 

The pretrained backbones are loaded with `from_pretrained`. They must therefore be available in the local Hugging Face cache or downloadable when the script is executed.

## Model-specific token alignment

The dataset stores one slot label per original word, while pretrained tokenizers may split a word into multiple sub-tokens. The implementation evaluates only one aligned sub-token per word:

- **BERT** uses the first sub-token because its encoder is bidirectional;
- **GPT-2** uses the last sub-token because its decoder is causal and the final piece has seen the complete word.

Padding positions, special tokens, and unused sub-tokens are assigned the ignore index `-100`, so they do not contribute to the slot-filling loss.

For intent classification:

- BERT uses the hidden state of `[CLS]`;
- GPT-2 appends `[EOS]` and uses the final non-padding hidden state.

## Evaluate the submitted checkpoints

Evaluation requires selecting exactly one model with `--model` and providing its checkpoint path.

Evaluate BERT:

```bash
python main.py --eval --model bert --checkpoint bin/bert_best_model.pt
```

Evaluate GPT-2:

```bash
python main.py --eval --model gpt2 --checkpoint bin/gpt2_best_model.pt
```

The script reads the pretrained model name from the checkpoint, rebuilds the correct tokenizer and model wrapper, verifies the slot and intent mappings, and prints test loss, slot F1, and intent accuracy.

## Train both pretrained models

Train the predefined BERT and GPT-2 configurations sequentially with:

```bash
python main.py --model both
```

This is the default behavior, so the following command is equivalent:

```bash
python main.py
```

## Train one pretrained model

Train only BERT:

```bash
python main.py --model bert
```

Train only GPT-2:

```bash
python main.py --model gpt2
```

By default, each selected model is fine-tuned across five runs with consecutive seeds. Checkpoints and summaries are written to:

```text
outputs/bert/
outputs/gpt2/
```

Each run checkpoint follows this naming scheme:

```text
<model_type>_run_<run_id>.pt
```

For submission, select the required checkpoint and copy it into `bin/` using a clear filename such as `bert_best_model.pt` or `gpt2_best_model.pt`.

## Main CLI arguments

| Argument | Purpose |
|---|---|
| `--model` | Select `bert`, `gpt2`, or `both`. Default: `both`. |
| `--eval` | Skip training and evaluate one saved checkpoint. |
| `--checkpoint` | Path to the checkpoint used with `--eval`. |
| `--dataset-dir` | ATIS dataset directory. Default: `dataset/ATIS`. |
| `--output-dir` | Root directory for generated training outputs. Default: `outputs`. |
| `--bert-model-name` | Hugging Face checkpoint used for BERT. Default: `google-bert/bert-base-uncased`. |
| `--gpt2-model-name` | Hugging Face checkpoint used for GPT-2. Default: `openai-community/gpt2`. |
| `--bert-lr` | BERT learning rate. Default: `5e-5`. |
| `--gpt2-lr` | GPT-2 learning rate. Default: `5e-5`. |
| `--dropout` | Dropout applied before the output heads. Default: `0.1`. |
| `--runs` | Number of repeated runs. Default: `5`. |
| `--seed` | Initial model and training-shuffle seed. |
| `--data-seed` | Fixed seed used to create the train/development split. |
| `--max-length` | Maximum tokenized sequence length. Default: `50`. |
| `--epochs` | Maximum number of training epochs. Default: `20`. |
| `--patience` | Early-stopping patience measured in validation checks. Default: `3`. |
| `--eval-every` | Number of epochs between development evaluations. Default: `1`. |

## Output summaries

After training, the script writes one summary file per selected model:

```text
outputs/bert/bert_summary.json
outputs/gpt2/gpt2_summary.json
```

Each summary records the test slot F1 and intent accuracy for every run together with their mean and standard deviation.
