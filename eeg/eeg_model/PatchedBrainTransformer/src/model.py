import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F


class LearningRateScheduler:
    """
    inspired by https://github.com/karpathy/nanoGPT/blob/master/train.py
    MIT License Copyright (c) 2022 Andrej Karpathy
    """

    def __init__(
        self, warmup_iters=150, learning_rate=3e-4, lr_decay_iters=1500, min_lr=3e-5
    ):
        self.warmup_iters = warmup_iters
        self.learning_rate = learning_rate
        self.lr_decay_iters = lr_decay_iters
        # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        self.min_lr = min_lr

    def get_lr(self, iteration):
        # Epochs starts with 0
        iteration += 1
        # 1) linear warmup for warmup_iters steps
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iteration > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iteration - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def state_dict(self):
        state_dict = {
            "warmup_iters": self.warmup_iters,
            "learning_rate": self.learning_rate,
            "lr_decay_iters": self.lr_decay_iters,
            "min_lr": self.min_lr,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict["warmup_iters"]
        self.learning_rate = state_dict["learning_rate"]
        self.lr_decay_iters = state_dict["lr_decay_iters"]
        self.min_lr = state_dict["min_lr"]


class LayerNorm(nn.Module):
    """
    LayerNorm with optional bias
    https://arxiv.org/pdf/1607.06450.pdf
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(
            x,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5,
        )


class MHSA(nn.Module):
    """
    Multi-Head Self-Attention block
    """

    def __init__(self, d_model, n_head, bias, dropout=0.0, flash_att=True):
        super().__init__()

        assert d_model % n_head == 0
        # key, query, value
        self.attn = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = dropout
        self.flash_att = flash_att

    def forward(self, x, split_sections=None):

        if split_sections is not None:
            x = torch.unsqueeze(input=x, dim=0)

        # batch size, sequence length, embedding dimensionality (d_model)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch
        q, k, v = self.attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        if split_sections is None:
            if self.flash_att:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=False,
                )
            else:
                y = (
                    self.attn_dropout(
                        F.softmax(
                            (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))),
                            dim=-1,
                        )
                    )
                    @ v
                )

            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, T, C)
        else:
            q = torch.tensor_split(q, split_sections, dim=2)
            k = torch.tensor_split(k, split_sections, dim=2)
            v = torch.tensor_split(v, split_sections, dim=2)

            if self.flash_att:
                att_dropout = self.dropout if self.training else 0
                # optimized by PyTorch 2.0
                y = torch.cat(
                    [
                        torch.nn.functional.scaled_dot_product_attention(
                            qs,
                            ks,
                            vs,
                            attn_mask=None,
                            dropout_p=att_dropout,
                            is_causal=False,
                        )
                        for qs, ks, vs in zip(q, k, v)
                    ],
                    dim=2,
                )

            else:
                y = torch.cat(
                    [
                        self.attn_dropout(
                            F.softmax(
                                (qs @ ks.transpose(-2, -1))
                                * (1.0 / math.sqrt(ks.size(-1))),
                                dim=-1,
                            )
                        )
                        @ vs
                        for qs, ks, vs in zip(q, k, v)
                    ],
                    dim=2,
                )

            # re-assemble all head outputs side by side
            y = y.transpose(1, 2).contiguous().view(B, T, C).squeeze(dim=0)

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    """
    Feed Forward block from Transformer
    """

    def __init__(self, d_model, dim_feedforward=None, dropout=0.0, bias=False):
        super().__init__()

        self.proj_in = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer model, based on 'Attention Is All You Need' -> https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, n_head, dropout=0.0, dim_feedforward=None, bias=False):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
            print(
                "dim_feedforward is set to 4*d_model, the default in Vaswani et al. (Attention is all you need)"
            )

        self.layer_norm_att = LayerNorm(d_model, bias=bias)
        self.mhsa = MHSA(d_model, n_head, bias, dropout=dropout, flash_att=True)
        self.layer_norm_ff = LayerNorm(d_model, bias=bias)
        self.feed_forward = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, bias=bias
        )

    def forward(self, x, split_sections):
        x = x + self.mhsa(self.layer_norm_att(x), split_sections)
        x = x + self.feed_forward(self.layer_norm_ff(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks, d_model, n_head, dropout, bias):
        super().__init__()

        self.encoder_block = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=dropout,
                    dim_feedforward=None,
                    bias=bias,
                )
                for _ in range(n_blocks)
            ]
        )

        # GPT2 type of init -> Radford et al. 'Language Models are Unsupervised Multitask Learners'
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_blocks))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, split_sections):
        for block in self.encoder_block:
            x = block(x, split_sections)
        return x


class PBT(nn.Module):
    def __init__(
        self,
        d_input,
        n_classes,
        num_embeddings,
        num_tokens_per_channel,
        d_model,
        n_blocks,
        num_heads,
        dropout,
        device,
        learnable_cls=False,
        bias_transformer=False,
        bert=False,
    ):
        super().__init__()

        self.num_tokens_per_channel = num_tokens_per_channel

        # linear projection layer, first layer in model
        self.linear_projection = nn.Linear(
            in_features=d_input, out_features=d_model, bias=False
        )

        if learnable_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.002)
        else:
            self.cls_token = torch.full(
                size=(1, 1, d_model),
                fill_value=0,
                requires_grad=False,
                dtype=torch.float32,
                device=device,
            )

        # trainable parameters for the position embedding
        # lookup table that stores learnable positional embedding
        self.pos_embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=d_model
        )

        self.transformer_encoder = TransformerEncoder(
            n_blocks=n_blocks,
            d_model=d_model,
            n_head=num_heads,
            dropout=dropout,
            bias=bias_transformer,
        )

        self.bert = bert
        if bert:
            self.linear_projection_out = nn.Linear(
                in_features=d_model, out_features=d_input, bias=False
            )

        self.cls_head = nn.Linear(
            in_features=d_model, out_features=n_classes, bias=True
        )

        # init all weights (linear_projection, cls_head )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.002)

    def forward(self, x, pos, split_sections=None):
        # Linear Projection, Concatenate [CLS]-Token, add positional embedding
        # x = torch.cat((self.cls_token.expand(x.size(0), 1, -1), self.linear_projection(x)), dim=1)
        x = self.linear_projection(x)

        if self.bert:
            x, pos_masking = self.masking(unmasked=x, probability_mask_token=0.3)

        # Transformer Encoder
        transformer_out = self.transformer_encoder(
            x + self.pos_embedding(pos), split_sections
        )

        if self.bert:
            logits = self.cls_head(transformer_out[:, 0])
            transformer_out = self.linear_projection_out(transformer_out[:, 1:])

            # MLP-Classifier, only [CLS]-Token is fed in
            return transformer_out, logits, pos_masking[:, 1:]
        else:
            if split_sections is None:
                return transformer_out, self.cls_head(transformer_out[:, 0]), None
            else:
                # MLP-Classifier, only [CLS]-Token is fed in
                return (
                    transformer_out,
                    self.cls_head(transformer_out[torch.where(pos == 0)[0]]),
                    None,
                )

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type, weight_decay_cls_head=0.0
    ):  
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py
        """
        
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        cls_head = param_dict["cls_head.weight"]
        del param_dict["cls_head.weight"]

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": [cls_head], "weight_decay": weight_decay_cls_head},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def masking(self, unmasked, probability_mask_token, copy=True):
        """
        BERT-style masking to input tensor along time dimension
        80% of masked token is replaced by (learnable) mask
        10% of masked token is replaced with random other token
        10% of masked token is untouched
        """
        if copy:
            x = unmasked.clone()
        else:
            x = unmasked

        # check if num_tokens_per_channel is int
        if (unmasked.size(1) - 1) / self.num_tokens_per_channel != (
            unmasked.size(1) - 1
        ) // self.num_tokens_per_channel:
            raise ValueError("num_channels is not a integer")
        num_channels = (unmasked.size(1) - 1) // self.num_tokens_per_channel

        # uniform distributed values in [0, 1) shape: Batch x num_tokens_per_channel
        pos_masking_rand = torch.rand((x.size(0), self.num_tokens_per_channel)).to(
            unmasked.device
        )

        # repeat masking for every EEG channel
        pos_masking_rand = pos_masking_rand.repeat(1, num_channels)

        # add [CLS]-Token that can never be masked
        pos_masking_rand = torch.cat(
            (
                torch.full(size=(x.size(0), 1), fill_value=2).to(unmasked.device),
                pos_masking_rand,
            ),
            dim=1,
        )

        # replace token in 80% with zero mask
        x[pos_masking_rand < probability_mask_token * 0.8] = torch.full(
            size=(x.size(-1),), fill_value=0.0
        ).to(unmasked.device)

        # replace token in 10% with random token from seq
        shuffled_token = x.clone()
        pos_random_token = torch.logical_and(
            pos_masking_rand >= probability_mask_token * 0.8,
            pos_masking_rand < probability_mask_token * 0.9,
        )
        x[pos_random_token] = shuffled_token.view(x.size(0) * x.size(1), x.size(2))[
            torch.randperm(x.size(0) * x.size(1))
        ].view(x.size())[pos_random_token]

        # 10% of token keep as it is
        pos_masking = pos_masking_rand < probability_mask_token

        return x, pos_masking

    @staticmethod
    def cos_sim_loss(output, target):
        """
        compares the angle of the output and target vectors
        see: https://en.wikipedia.org/wiki/Cosine_similarity
        out: mean(1 - cos_sim(output, target))
        """

        cos_sim = nn.CosineSimilarity(dim=1)

        return (
            torch.full((output.size(0),), 1).to(output.device)
            - cos_sim(
                torch.flatten(output, start_dim=1), torch.flatten(target, start_dim=1)
            )
        ).mean()
