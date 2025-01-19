import torch
from torch import nn, distributions
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from gpt2 import GPT2ModelParams


class Policy(nn.Module):
    def __init__(
            self,
            lm_model: GPT2LMHeadModel,  # language model
            tokenizer: GPT2Tokenizer,
            config: dict,
            training: bool
    ):
        super().__init__()
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.config = config

        if not training:
            for param in self.lm_model.parameters():
                param.requires_grad_(False)

    def forward(self, input_token, attention_mask):
        """
        :param input_token: shape=(batch_size * 2, query_length + response_length -1)
        :return: language head output logits, i.e., raw probabilities. shape=(batch_size * 2, response_length, vocab_size)
        """
        output = self.lm_model(input_token, attention_mask=attention_mask)
        return output.logits

    def loss(self, policy_ref, input_ids, win_idx, lose_idx):
        """
        :param policy_ref: reference policy model
        :param input_ids: batched query + responses. shape=(batch_size * 2, query_length + response_length)
        :param win_idx: index of query + preferred_response. shape=(batch_size, )
        :param lose_idx: index of query + non-preferred_response. shape=(batch_size, )
        :return: DPO loss
        """
        query_length = self.config['task']['query_length']
        input_token, label = input_ids[:, :-1], input_ids[:, query_length:].clone()  # label.shape=(batch_size * 2, response_length)

        # OpenAI GPT-2 uses vocab_size + 2, i.e., 50259 as padding token, which is not acceptable by hugging face transformers
        # Follow DPO's implementation, https://github.com/eric-mitchell/direct-preference-optimization,
        # Here use 0 for padding input, and -100 for padding label
        pad_token_id = self.config['openai_gpt2_pad_token_id']
        attention_mask = torch.ne(input_token, pad_token_id).to(device=self.config['device'])
        input_token.masked_fill_(input_token == pad_token_id, 0)
        label_mask = torch.ne(label, pad_token_id).to(device=self.config['device'])
        label.masked_fill_(label == pad_token_id, -100)

        (n, response_length), query_length = label.shape, self.config['task']['query_length']
        with torch.no_grad():
            logits_ref = policy_ref(input_token, attention_mask)[:, query_length-1:, :]  # shape=(batch_size * 2, response_length, vocab_size)
            logp_ref = -F.cross_entropy(logits_ref.reshape(n * response_length, -1), label.reshape(-1).long(), reduction='none')  # shape=(n * response_length,)
            logp_ref = torch.sum(logp_ref.reshape(n, response_length) * label_mask, dim=-1)  # ignore logp on the padding tokens

        logits = self(input_token, attention_mask)[:, query_length-1:, :]
        logp = -F.cross_entropy(logits.reshape(n * response_length, -1), label.reshape(-1).long(), reduction='none')  # shape=(n * response_length,)
        logp = torch.sum(logp.reshape(n, response_length) * label_mask, dim=-1)  # ignore logp on the padding tokens. logp.shape=(batch_size * 2,)

        logp_ref_w, logp_ref_l = logp_ref[win_idx], logp_ref[lose_idx]
        logp_w, logp_l = logp[win_idx], logp[lose_idx]
        loss_logits = (logp_w - logp_l) - (logp_ref_w - logp_ref_l)
        beta, label_smoothing = self.config['beta'], self.config['label_smoothing']
        loss = -F.logsigmoid(beta * loss_logits) * (1 - label_smoothing) - F.logsigmoid(-beta * loss_logits) * label_smoothing
        return loss.mean()

    @torch.no_grad()
    def sample(self, query, max_length):
        pad_token_id = self.config['openai_gpt2_pad_token_id']
        attention_mask = torch.ne(query, pad_token_id).to(device=self.config['device'])
        query.masked_fill_(query == pad_token_id, 0)  # see comment in the loss() method.
        max_context_length = GPT2ModelParams.n_position
        logits_scale = 1 / torch.tensor(self.config['task']['temperature'], dtype=torch.float32, device=self.config['device'])

        for _ in range(max_length):
            lm_logits = self(query, attention_mask)[:, -1, :] * logits_scale
            dist = distributions.Categorical(logits=lm_logits)
            next_token = dist.sample()
            query = torch.cat((query, next_token[..., None]), dim=-1)[-max_context_length:]
            new_mask = torch.ne(next_token, 0)[..., None]
            attention_mask = torch.cat((attention_mask, new_mask), dim=-1)

        return query
