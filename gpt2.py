from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Config


@dataclass
class GPT2ModelParams:
    vocab_size: int = 50257  # do not change
    n_position: int = 1024  # block_size

    # gpt2 configs. Do not change. leave here for documentation purpose.
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    # dropout probs
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0


def get_model(config: dict) -> GPT2LMHeadModel:
    mcfg = GPT2Config()
    for k, v in asdict(GPT2ModelParams()).items():
        setattr(mcfg, k, v)

    if 'ckpt' in config:
        if not Path(config['ckpt']).is_file():
            raise ValueError(f"checkpoint does not exist: {config['ckpt']}")
        print(f"Load model {config['model']} from checkpoint {config['ckpt']}")
        ckpt = torch.load(config['ckpt'])
        model = GPT2LMHeadModel(mcfg)
        model.load_state_dict(ckpt.pop('model'))
    else:
        print(f"Load a pretrained model {config['model']} from hugging face")
        model = GPT2LMHeadModel.from_pretrained(config['model'])
    return model


if __name__ == '__main__':
    config = {'model': 'gpt2', 'ckpt': 'model_124M/124M_ckpt.pt'}
    print(get_model(config))

