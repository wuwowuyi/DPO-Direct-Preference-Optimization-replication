import torch
from torch import distributions
from torch.nn import functional as F
from transformers import GPT2Tokenizer

from gpt2 import GPT2ModelParams


class Policy:
    def __init__(
            self,
            lm_model,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            train: bool,
            device: int = 0,  # index of current device
            task: str = 'sentiment'  # name of task
    ):
        super().__init__()
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.task = task

        if not train:
            for param in self.lm_model.parameters():
                param.requires_grad_(False)

    def log_prob(self, input_ids):
        """
        :param input_ids: shape=(batch_size * 2, query_length + response_length)
        :return:
        """
        query_length = self.config[self.task]['query_length']
        input_token = input_ids[:, :-1].clone()  # shape=(batch_size * 2, query_length + response_length - 1)
        label = input_ids[:, query_length:].clone()  # shape=(batch_size * 2, response_length)

        pad_token_id = self.config['openai_gpt2_pad_token_id']
        attention_mask = torch.ne(input_token, pad_token_id).to(device=self.device)
        input_token.masked_fill_(input_token == pad_token_id, self.tokenizer.pad_token_id)
        label_mask = torch.ne(label, pad_token_id).to(device=self.device)
        label.masked_fill_(label == pad_token_id, self.tokenizer.pad_token_id)

        n, response_length = label.shape
        logits = self.lm_model(input_token, attention_mask=attention_mask).logits
        logits = logits[:, query_length - 1:, :]  # shape=(batch_size * 2, response_length, vocab_size)
        logp = -F.cross_entropy(logits.reshape(n * response_length, -1), label.reshape(-1).long(), reduction='none')  # shape=(n * response_length,)
        logp = torch.sum(logp.reshape(n, response_length) * label_mask, dim=-1)  # ignore logp on the padding tokens
        return logp

    def loss(self, policy_ref, input_ids, win_idx, lose_idx):
        """
        :param policy_ref: reference policy model
        :param input_ids: batched query + responses. shape=(batch_size * 2, query_length + response_length)
        :param win_idx: index of query + preferred_response. shape=(batch_size, )
        :param lose_idx: index of query + non-preferred_response. shape=(batch_size, )
        :return: DPO loss
        """
        with torch.no_grad():
            logp_ref = policy_ref.log_prob(input_ids)

        logp = self.log_prob(input_ids)

        logp_ref_w, logp_ref_l = logp_ref[win_idx], logp_ref[lose_idx]
        logp_w, logp_l = logp[win_idx], logp[lose_idx]
        loss_logits = (logp_w - logp_l) - (logp_ref_w - logp_ref_l)
        beta, label_smoothing = self.config['beta'], self.config['label_smoothing']
        loss = -F.logsigmoid(beta * loss_logits) * (1 - label_smoothing) - F.logsigmoid(-beta * loss_logits) * label_smoothing
        return loss.mean()

    @torch.no_grad()
    def sample(self, query, max_length):
        pad_token_id = self.config['openai_gpt2_pad_token_id']
        attention_mask = torch.ne(query, pad_token_id).to(device=self.device)
        query.masked_fill_(query == pad_token_id, self.tokenizer.pad_token_id)
        max_context_length = GPT2ModelParams.n_positions
        logits_scale = 1 / torch.tensor(self.config[self.task]['temperature'], dtype=torch.bfloat16, device=self.device)

        for _ in range(max_length):
            logits = self.lm_model(query, attention_mask=attention_mask).logits
            lm_logits = logits[:, -1, :] * logits_scale
            dist = distributions.Categorical(logits=lm_logits)
            next_token = dist.sample()
            query = torch.cat((query, next_token[..., None]), dim=-1)[-max_context_length:]
            new_mask = torch.ones_like(next_token, device=self.device)[..., None]
            attention_mask = torch.cat((attention_mask, new_mask), dim=-1)

        return query
