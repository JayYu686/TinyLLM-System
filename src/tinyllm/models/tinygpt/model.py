"""Decoder-only TinyGPT causal language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from tinyllm.models.tinygpt.config import TinyGPTConfig
from tinyllm.models.tinygpt.layers import RMSNorm, TransformerBlock


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """Output of a TinyGPT causal language model forward pass."""

    logits: Tensor
    loss: Tensor | None = None


class TinyGPT(nn.Module):
    """Small decoder-only transformer used to validate training infrastructure."""

    def __init__(self, config: TinyGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_module)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        """Compute logits and an optional next-token causal language-model loss."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long dtype")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds model.max_sequence_length")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain token IDs outside model vocabulary")
        if labels is not None and (labels.shape != input_ids.shape or labels.dtype != torch.long):
            raise ValueError("labels must match input_ids shape and use torch.long dtype")

        hidden_states = self.embedding_dropout(self.token_embeddings(input_ids))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        logits = self.lm_head(self.final_norm(hidden_states))

        loss: Tensor | None = None
        if labels is not None:
            if labels.shape[1] < 2:
                raise ValueError("at least two tokens are required to compute causal LM loss")
            loss = functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        """Return the number of unique model parameters."""

        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)
