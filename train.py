import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda import nccl

import wandb
import yaml
from torch import optim
from tqdm import tqdm
from transformers import GPT2Tokenizer

from gpt2 import get_model, GPT2ModelParams
from label import download_labels, LabelBuffer
from policy import Policy


def train(config: dict):

    seed = 1337 + config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config['device']

    # prepare human labelled data
    labels = download_labels(config['task']['name'])
    data_buffer = LabelBuffer(labels)

    # load model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config['openai_gpt2_pad_token_id'] = 50259  # padding token id used in OpenAI's human labelled dataset

    policy_ref = Policy(get_model(config), tokenizer, config, False)
    policy_ref.eval()
    policy_ref.to(device)

    policy = Policy(get_model(config), tokenizer, config, True)
    policy.train()
    policy.to(device)

    optimizer = optim.Adam(policy.parameters(), lr=config['lr'])

    # training, and evaluation.
    # train one epoch
    total, batch_size = len(data_buffer), config['batch_size']

    def evaluation(before_training: bool = False):
        def _print_sample(samples: torch.Tensor):
            for idx, s in enumerate(samples):
                print(f"\nthe {idx}-th sample:")
                print(tokenizer.decode(s))

        queries, preferred_response = data_buffer.get_eval_query_response()
        max_length = config['task']['response_length']

        if before_training:
            queries_clone = queries.clone()
            pad_token_id = config['openai_gpt2_pad_token_id']
            queries_clone.masked_fill_(queries == pad_token_id, 220)  # 220 is empty space ' '
            preferred_response.masked_fill_(preferred_response == pad_token_id, 220)
            print("*" * 50)
            print('User preferred query response from training data')
            _print_sample(torch.cat((queries_clone, preferred_response), dim=-1))

            policy_ref_response = policy_ref.sample(queries.clone().to(device), max_length)
            print("\nSamples from the reference policy model:")
            _print_sample(policy_ref_response)
        else:
            policy_response = policy.sample(queries.to(device), max_length)
            print("\nSamples from the policy model:")
            _print_sample(policy_response)


    for i in range(config['epoch']):
        evaluation(True)

        print(f'Start training epoch {i}')
        order = torch.randperm(total)
        for j in tqdm(range(total // batch_size)):
            idx = order[j * batch_size: j * batch_size + batch_size]
            batch, win_idx, lose_idx = data_buffer.get_batch(idx)
            batch, win_idx, lose_idx = batch.to(device), win_idx.to(device), lose_idx.to(device)

            loss = policy.loss(policy_ref, batch, win_idx, lose_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "loss": loss,
            })

        evaluation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    if not Path(args.config_file).is_file():
        raise ValueError(f"Cannot find configuration file: {args.config_file}")
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # GPU and communication support
    support_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and nccl.version() >= (2, 10)
    if not support_bf16:
        print("Must install GPUs that support bfloat16.")
        sys.exit(0)
    config['device'] = 'cuda'

    config['seed'] = args.seed

    if args.wandb_log:  # wandb logging
        wandb_project = 'dpo'
        wandb_run_name = str(int(time.time()))
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    train(config)

if __name__ == '__main__':
    main()
