import argparse
import functools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import distributed, nn
from torch import optim
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from tqdm import tqdm
from transformers import GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import wandb
from checkpoint_handler import save_model_checkpoint, save_model_and_optimizer_sharded
from gpt2 import get_model
from label import LabelBuffer, download_labels
from policy import Policy


'''Train models on multiple GPUs using FSDP. '''


def setup():
    # initialize the process group
    distributed.init_process_group("nccl")


def cleanup():
    distributed.destroy_process_group()


def check_fn(submodule: nn.Module) -> bool:
    """will be passed each child submodule and returns
            `True` or `False` depending on whether the submodule should be wrapped."""
    return isinstance(submodule, GPT2Block)  # same as wrapping


def train(config: dict):
    seed = 1337 + config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # FSDP environment variables and setup
    local_rank = int(os.environ['LOCAL_RANK'])  # rank on local node
    rank = int(os.environ['RANK'])  # global rank
    world_size = int(os.environ['WORLD_SIZE'])  # total number of devices
    setup()
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    if config['wandb_log'] and rank == 0:  # wandb logging
        wandb_project = 'dpo'
        wandb_run_name = str(int(time.time()))
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # prepare human labelled data
    task = config['task']
    labels = download_labels(config[task]['label'], download=(local_rank == 0))
    data_buffer = LabelBuffer(labels)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    config['openai_gpt2_pad_token_id'] = 50259  # padding token id used in OpenAI's human labelled dataset
    tokenizer.pad_token_id = tokenizer.eos_token_id

    gpt2_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block})
    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,  # Gradient communication precision.
        buffer_dtype=torch.bfloat16  # Buffer precision.
    )

    policy_ref = Policy(FSDP(get_model(config),
                   auto_wrap_policy=gpt2_auto_wrap_policy,
                   mixed_precision=bfSixteen,
                   use_orig_params=True,
                   device_id=device), tokenizer, config, False, device, task)
    policy_ref.lm_model.eval()

    policy = Policy(FSDP(get_model(config),
                   auto_wrap_policy=gpt2_auto_wrap_policy,
                   mixed_precision=bfSixteen,
                   use_orig_params=True,
                   device_id=device), tokenizer, config, True, device, task)
    if 'activation_checkpointing' in config and config['activation_checkpointing']:
        apply_activation_checkpointing(policy.lm_model, check_fn=check_fn)
    policy.lm_model.train()

    # prepare training
    optimizer = optim.Adam(policy.lm_model.parameters(), lr=config[task]['lr'])
    local_total, local_batch_size = len(data_buffer) // world_size, config[task]['batch_size'] // world_size  # TODO

    def evaluation(before_training: bool = False):
        def _print_sample(samples: torch.Tensor):
            for ith, s in enumerate(samples):
                print(f"\nthe {ith}-th sample:")
                print(tokenizer.decode(s))

        queries, preferred_response = data_buffer.get_eval_query_response(10)
        max_length = config[task]['response_length']

        if before_training:
            queries_clone = queries.clone()
            pad_token_id = config['openai_gpt2_pad_token_id']
            queries_clone.masked_fill_(queries == pad_token_id, tokenizer.pad_token_id)  # 220 is empty space ' '
            preferred_response.masked_fill_(preferred_response == pad_token_id, tokenizer.pad_token_id)
            print("*" * 50)
            print('User preferred query response from training data')
            _print_sample(torch.cat((queries_clone, preferred_response), dim=-1))

            policy_ref_response = policy_ref.sample(queries.to(device), max_length)
            print("\nSamples from the reference policy model:")
            _print_sample(policy_ref_response)
        else:
            policy_response = policy.sample(queries.to(device), max_length)
            print("\nSamples from the policy model:")
            _print_sample(policy_response)

    for i in range(config[task]['epoch']):
        evaluation(True)

        print(f'Start training epoch {i} on rank {rank}')
        order = rank * local_total + torch.randperm(local_total)  # or use distributed.all_reduce to get a global permutation.
        for j in tqdm(range(local_total // local_batch_size), desc=f"epoch-{i} on rank-{rank}"):
            idx = order[j * local_batch_size: j * local_batch_size + local_batch_size]
            batch, win_idx, lose_idx = data_buffer.get_batch(idx)
            batch, win_idx, lose_idx = batch.to(device), win_idx.to(device), lose_idx.to(device)

            loss = policy.loss(policy_ref, batch, win_idx, lose_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            fsdp_loss = loss.clone().detach()
            distributed.all_reduce(fsdp_loss, op=distributed.ReduceOp.SUM)
            if config['wandb_log'] and rank == 0:
                if config['log_local_loss']:
                    wandb.log({
                        f"loss_{rank}": loss,
                    })

                wandb.log({
                    f"loss": fsdp_loss / world_size,
                })

    evaluation(False)

    if config['checkpoint_type'] == 'FULL_STATE_DICT':
        save_model_checkpoint(
            policy.lm_model, optimizer, rank, config, epoch=1
        )
    elif config['checkpoint_type'] == 'SHARDED_STATE_DICT':
        save_model_and_optimizer_sharded(policy.lm_model, rank, config)

    distributed.barrier()
    cleanup()


# launch using `torchrun --nnodes <num_node> --nproc_per_node <num_gpu>  T5_training.py`.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, default="config/gpt2_small_hf.yaml")
    parser.add_argument("--task", type=str, default="sentiment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_log", action="store_true")
    args = parser.parse_args()

    if not Path(args.config_file).is_file():
        raise ValueError(f"Cannot find configuration file: {args.config_file}")
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # GPU and communication support
    support_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and torch.cuda.nccl.version() >= (2, 10)
    if not support_bf16:
        print("Must install GPUs that support bfloat16.")
        sys.exit(0)

    config['seed'] = args.seed
    config['wandb_log'] = args.wandb_log
    config['task'] = args.task

    print(config)

    train(config)


if __name__ == '__main__':
    main()
