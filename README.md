
# StARformer
This repository contains the PyTorch implementation for our paper titled StARformer: Transformer with State-Action-Reward Representations.
We learn local State-Action-Reward representations (StAR-representations) to improve (long) sequence modeling for reinforcement learning (and imitation learning).

## Installation

Dependencies can be installed by Conda:

```
conda env create -f my_env.yml
```

And [install Atari ROMs](https://github.com/openai/atari-py#roms).

## Downloading datasets
Please follow [Decision-Transformer](https://github.com/kzl/decision-transformer/blob/master/atari/readme-atari.md)


## Example usage

See `run.sh` or below:
```
python run_star_atari.py --seed 123 --data_dir_prefix [data_directory] --epochs 10 --num_steps 500000 --num_buffers 50 --batch_size 64 --seq_len 30 --model_type 'star' --game 'Breakout'
```
`[data_directory]` is where you place the Atari dataset.

Variants (`model_type`):
 - `'star'` (imitation)
 - `'star_rwd'` (offline RL)


## Acknowledgement

This code is based on [Decision-Transformer](https://github.com/kzl/decision-transformer/).
