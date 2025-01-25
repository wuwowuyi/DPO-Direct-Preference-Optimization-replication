
My replication of the [DPO paper](https://arxiv.org/abs/2305.18290), implemented from scratch using Pytorch, trained with FSDP. (in progress)

## Setup
### Environment
* For simplicity, only runs on GPUs that support bfloat16, i.e., Ampere or newer. Alternatively, for GPUs not supporting bfloat16, we can use float16 + torch.amp package (mixed precision) + gradient scaling, which would use more GPU memory though.
* For simplicity, when FSDP is enabled we use `torchrun` to launch training, this requires CPU big enough to fit the entire model since only later FSDP will shard the model on multiple GPUs. Alternatively, for very large models, we can also use Hugging Face's `device_map` to load models onto GPUs and CPU, and then use `torch.multiprocessing.spawn()` to launch training. See [Loading big models into memory](https://huggingface.co/docs/accelerate/en/concept_guides/big_model_inference#the-devicemap)
* Python 3.9+ 

Tested on pretrained GPT-2 models hosted by Hugging Face.

| model  | #params | n_layer | n_head | n_embd |
|--------|---------|---------|--------|--------|
| small  | 124M    | 12      | 12     | 768    |
| large  | 774M    | 36      | 20     | 1280   |
| xl     | 1.5B    | 48      | 25     | 1600   |

### Train

To train and log, run for example `torchrun --nnodes 1 --nproc_per_node 2  train_fsdp.py --config_file config/gpt2_small_hf.yaml --wandb_log`

## Evaluation
Training datasets are borrowed from [RLHF training](https://github.com/wuwowuyi/Fine-Tuning-Language-Models-from-Human-Preferences-Pytorch-Implementation).

### Sentiment continuation 
This is a stylistic continuation task, and model is encouraged to generate a **"positive and happy" continuation** of a query which is sampled from a training dataset, like an excerpt from the BookCorpus dataset.

Training dataset: OpenAI's sentiment offline_5k.json.

Training hyperparameters: 1 epoch, learning rate 5e-5, sampling temperature 0.7. query length 64, response length 24. 

#### GPT2 small

| query                                                                                                                                                                                                                                                                                            |  Before training                                                          | DPO trained                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |-------------------------------------------------------------------------------------|
| He did not like things which reminded him that he had once been not only outside the progressive element but even outside the College. He did not always like Curry either. His pleasure in being with him was not that sort of pleasure.<br>"Yes," said Curry. "Denniston was your chief rival. | This time you were doing something different. He was not so much a rival that he'd be matched.           | An honor to be with him was really rewarding. I had an amazing time working with him.|
| Corolan moved to the next chair down, giving me his. Wylend floated my water glass across the table. Water was the only thing Wyatt would allow me to drink. I'd already had my broth.                                                                                                           | " (I'm okay with that. I should give it to him.)"        | Ever since we'd gotten married, we held hands all year round. Then we started to talk about my future plans.|
| She took too many hits in the head playing football and soccer and they ruined her mind. She has fighter's dementia, similar to what the Quarry brothers had. Do you want to be that way, dear?"<br>Sean blinked. He hadn't known why Ruby was always shaking. He was silent for a moment.       | He had never been to school, he never had been to school, he was a kid who was looking forward to school|He felt the tears drop from his eyes. "Well, I would love to be that way," she said|

#### GPT2 Large
Queries are the same as GPT2 small.

| Response before training | DPO trained |
|-----------------------|-------------|
|This time you have to get over it." "And I want to get over it with you," said D|An interesting man he was." "Well, good for you. He was a good fighter.|
|When I'd finished, I went to get the rest of the stew.|Ever since we got Wylend, he'd been kind to me. He was always willing to help me cook things|
|"Ruby, I know you're worried about me. You always have, and I know you want me to be|"Good. That means you're listening." She smiled. "Thank you."|

