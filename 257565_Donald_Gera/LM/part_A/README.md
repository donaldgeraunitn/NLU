## Dependencies

Install the dependencies declared in the global `requirements.txt` file from the root of the submitted project:

```bash
pip install -r requirements.txt
```

For this part, the directly used packages are `torch`, `transformers`, `numpy`, `matplotlib`, and `tqdm`.

## Evaluate the submitted best model

The submitted checkpoint is stored in:

```text
bin/best_model.pt
```

Evaluate it on the Penn Treebank test split with:

```bash
python main.py --eval --checkpoint bin/best_model.pt
```

The checkpoint stores the model class and architecture configuration. The evaluation path rebuilds the correct variant automatically before loading its parameters.

## Reproduce training experiments

Training creates the corresponding `outputs/` directories automatically. Run the predefined stages sequentially because later stages reuse reports produced by earlier stages:

```bash
python main.py --experiment baseline_lr
python main.py --experiment greedy_hparams
python main.py --experiment final_lr
python main.py --experiment dropout
python main.py --experiment weight_tying
```

Supported `--experiment` values:

- `baseline_lr`: tune the learning rate for the fixed baseline configuration;
- `greedy_hparams`: tune `d_model`, `n_heads`, `num_layers`, and `ff_dim` sequentially;
- `final_lr`: readjust the learning rate for the selected architecture;
- `dropout`: tune the dropout probability;
- `weight_tying`: enable weight tying and tune its learning rate.

The code does not expose a dedicated run-all flag. The complete predefined procedure is the sequence of commands shown above.

## Outputs

Unless `--output_dir` is specified, each training stage stores `best_model.pt`, `report.json`, `report.txt`, and plots in its own directory:

```text
outputs/baseline_lr_tuning/
outputs/greedy_hparams/
outputs/final_lr_readjustment/
outputs/dropout_tuning/
outputs/weight_tying/
```

## Useful CLI arguments

Candidate lists can be overridden without modifying the code:

```bash
python main.py --experiment baseline_lr --learning_rates 0.001 0.002 0.003 0.005
python main.py --experiment greedy_hparams --d_model_values 20 40 80 160 --n_heads_values 1 2 4 8 --num_layers_values 1 2 3 4 --ff_dim_values 20 40 80 160 320
python main.py --experiment dropout --dropout_values 0.1 0.2 0.3 0.5
```

Other supported options include `--learning_rate`, `--train_batch_size`, `--eval_batch_size`, `--n_epochs`, `--patience`, `--weight_decay`, `--grad_clip`, `--seed`, `--device`, `--tokenizer_name`, and `--output_dir`. The input reports used by later stages can be overridden with `--baseline_report`, `--hparams_report`, `--final_lr_report`, and `--dropout_report`.