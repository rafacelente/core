from typing import Optional, cast
import math

import torch
import torch.nn as nn

from core.config import AttentionConfig, DType, FeedForwardConfig, LayerNormConfig
from core.modules.attention import NormalizedAttention, Attention
from core.modules.block import NormalizedBlock, Block
from core.modules.init import InitMethod
from core.utils import BufferCache, get_default_device
from core.models.model_utils import CoreOutput


class CoreModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        attention_config: AttentionConfig,
        feed_forward_config: FeedForwardConfig,
        layer_norm_config: LayerNormConfig,
        dropout: float = 0.0,
        dtype: DType = DType.FLOAT32,
        init_method: InitMethod = InitMethod.NORMAL,
        init_seed: int = 42,
        loss_fn: Optional[nn.Module] = None,
        ignore_index: int = -100,
    ):
        super().__init__()

        cache = BufferCache()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.attention_config = attention_config
        self.feed_forward_config = feed_forward_config
        self.layer_norm_config = layer_norm_config
        self.dropout = dropout
        self.dtype = dtype
        self.ignore_index = ignore_index
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.embeddings = nn.Embedding(vocab_size, d_model, dtype=dtype.to_torch_dtype())
        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            block = Block(
                d_model=d_model,
                block_idx=block_idx,
                attention_config=attention_config,
                feed_forward_config=feed_forward_config,
                layer_norm_config=layer_norm_config,
                cache=cache,
            )
            self.blocks[str(block_idx)] = block
        self.lm_head = nn.Linear(d_model, vocab_size, dtype=dtype.to_torch_dtype(), bias=False)

        self.init_method = InitMethod(init_method)
        self.init_seed = torch.Generator().manual_seed(init_seed)

        self._cache = cache
        self._device: Optional[torch.device] = None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for p in self.parameters():
                if p.numel() > 0:
                    self._device = p.device
                    break
            if self._device is None:
                self._device = get_default_device()
        return self._device

    @torch.no_grad()
    def init_weights(
        self,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = self.device

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.embeddings is not None:
            self.init_method.init_embeddings(self.embeddings, generator=self.init_seed)
        if self.lm_head is not None:
            self.init_method.init_final_w_out(self.lm_head, d_model=self.d_model, generator=self.init_seed)

        for block in self.blocks.values():
            block = cast(Block, block)
            att = cast(Attention, block.attention)
            self.init_method.init_attention(att, self.n_layers, generator=self.init_seed)
            self.init_method.init_feed_forward(
                block.feed_forward, self.d_model, self.n_layers, generator=self.init_seed
            )

            if att.use_rope and max_seq_len is not None:
                att.rope.warmup_cache(max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CoreOutput:
        x = self.embeddings(input_ids)
        for block in self.blocks.values():
            x = block(x)
        logits = self.lm_head(x)
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return CoreOutput(logits=logits, loss=loss)
        return CoreOutput(logits=logits)

class NormalizedCoreModel(CoreModel):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        vocab_size: int,
        attention_config: AttentionConfig,
        feed_forward_config: FeedForwardConfig,
        layer_norm_config: LayerNormConfig,
        dropout: float = 0.0,
        dtype: DType = DType.FLOAT32,
        init_method: InitMethod = InitMethod.NORMALIZED,
        init_seed: int = 42,
        loss_fn: Optional[nn.Module] = None,
        ignore_index: int = -100,
    ):
        super().__init__(d_model, n_layers, vocab_size, attention_config, feed_forward_config, layer_norm_config, dropout, dtype, init_method, init_seed, loss_fn, ignore_index)
        if self.dropout > 0.0:
            raise ValueError("NormalizedCoreModel does not support dropout")
        del self.blocks
        self.blocks = nn.ModuleDict()
        for block_idx in range(n_layers):
            block = NormalizedBlock(
                d_model=d_model,
                block_idx=block_idx,
                attention_config=attention_config,
                feed_forward_config=feed_forward_config,
                layer_norm_config=layer_norm_config,
                cache=self._cache,
            )
            self.blocks[str(block_idx)] = block
        self.sz_init_value = 1.0
        self.sz_init_scaling = 1.0 / math.sqrt(d_model)
        self.sz = nn.Parameter(self.sz_init_scaling * torch.ones(vocab_size))

    def init_weights(self, max_seq_len: Optional[int] = None, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.device

        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        if self.embeddings is not None:
            self.init_method.init_embeddings(self.embeddings, std=self.d_model**-0.5, generator=self.init_seed)
        if self.lm_head is not None:
            self.init_method.init_final_w_out(self.lm_head, d_model=self.d_model, std=self.d_model**-0.5, generator=self.init_seed)

        for block in self.blocks.values():
            block = cast(NormalizedBlock, block)
            att = cast(NormalizedAttention, block.attention)
            self.init_method.init_attention(att, self.n_layers, generator=self.init_seed)
            self.init_method.init_feed_forward(
                block.feed_forward, self.d_model, self.n_layers, generator=self.init_seed
            )

            if att.use_rope and max_seq_len is not None:
                att.rope.warmup_cache(max_seq_len=max_seq_len, device=device)

        self.normalize_matrices()

    def justnorm(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / x.norm(p=2, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)

    def _normalize_matrix(self, m: torch.Tensor, dim: int = -1) -> torch.Tensor:
        m.copy_(self.justnorm(m, dim=dim))

    @torch.no_grad()
    def normalize_matrices(self) -> None:
        if self.embeddings is not None:
            self._normalize_matrix(self.embeddings.weight)
        if self.lm_head is not None:
            self._normalize_matrix(self.lm_head.weight)

        for block in self.blocks.values():
            block = cast(NormalizedBlock, block)
            att = cast(NormalizedAttention, block.attention)
            ff = block.feed_forward
            att.normalize_matrices()
            ff.normalize_matrices()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CoreOutput:
        x = self.embeddings(input_ids)
        for block in self.blocks.values():
            x = block(x)
        logits = self.lm_head(x)
        sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
        logits = sz * logits
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return CoreOutput(logits=logits, loss=loss)
        return CoreOutput(logits=logits)

    def post_optim_step(self) -> None:
        self.normalize_matrices()