import os
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

CACHE_DIR = os.getenv("HF_CACHE_DIR", "./cache")

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
            Multi-head self-attention layer for transformer models.
            
            This layer implements scaled dot-product attention with multiple heads,
            allowing the model to attend to information from different representation
            subspaces at different positions.
            
            Attributes:
                c_attn (nn.Linear): Combined linear projection for query, key, and value.
                    Projects from embedding dimension to 3 times the embedding dimension
                    to generate Q, K, V in a single batch operation.
                c_proj (nn.Linear): Output projection layer that projects the concatenated
                    multi-head attention output back to the embedding dimension.
                n_head (int): Number of attention heads.
                n_embd (int): Embedding dimension. Must be divisible by n_head.
                bias (torch.Tensor): Lower triangular mask for causal (autoregressive) attention.
                    Prevents positions from attending to future positions.
            
            Args:
                config: Configuration object containing:
                    - n_embd (int): Embedding dimension
                    - n_head (int): Number of attention heads
                    - block_size (int): Maximum sequence length
            """
    def __init__(self, config):
            
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # attention (materializes the large (T,T) matrix for all the queries and keys)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
        
        present = (k, v)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :k.size(-2)] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.dropout(self.c_proj(y))
        return y, present

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module for transformer models.
    This module implements a feed-forward network commonly used in transformer architectures.
    It consists of two linear transformations with a GELU activation in between,
    expanding the embedding dimension by a factor of 4 and then projecting back.
    The MLP follows the architecture: Linear(n_embd -> 4*n_embd) -> GELU -> Linear(4*n_embd -> n_embd)
    Attributes:
        c_fc (nn.Linear): First linear layer that expands the embedding dimension.
        gelu (nn.GELU): GELU activation function with tanh approximation.
        c_proj (nn.Linear): Projection layer that reduces back to original embedding dimension.
    Args:
        config: Configuration object containing:
            - n_embd (int): Embedding dimension size.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    Transformer block that consists of a multi-head self-attention layer and a feed-forward network (MLP).
    
    This block applies layer normalization, followed by the attention mechanism and the MLP,
    with residual connections around each of these components.

    Attributes:
        ln_1 (nn.LayerNorm): Layer normalization applied before the attention mechanism.
        attn (CausalSelfAttention): The causal self-attention layer.
        ln_2 (nn.LayerNorm): Layer normalization applied before the MLP.
        mlp (MLP): The multi-layer perceptron used for feed-forward processing.
    
    Args:
        config: Configuration object containing:
            - n_embd (int): Embedding dimension size.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        

    def forward(self, x, past_kv=None):
        attn_out, present = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present

@dataclass
class GPT2Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    dropout: float = 0.1

class GPT2(nn.Module):
    """
    Generative Pre-trained Transformer (GPT) model.

    This model implements the GPT architecture, which consists of an embedding layer,
    multiple transformer blocks, and a final linear layer for language modeling.
    
    Attributes:
        config (GPTConfig): Configuration object containing model parameters.
        transformer (nn.ModuleDict): Dictionary containing the embedding layers and transformer blocks.
        lm_head (nn.Linear): Linear layer for generating output logits from the final hidden states.
    
    Args:
        config: Configuration object containing:
            - vocab_size (int): Size of the vocabulary.
            - block_size (int): Maximum sequence length.
            - n_layer (int): Number of transformer blocks.
            - n_head (int): Number of attention heads.
            - n_embd (int): Embedding dimension size.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        # forward the token and posisition embeddings
        if past_kvs is None:
            past_length = 0
            past_kvs = [None] * len(self.transformer.h)
        else:
            past_length = 0 if past_kvs[0] is None else past_kvs[0][0].size(-2)
        assert past_length + T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(past_length, past_length + T, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # apply dropout to the sum of token and position embeddings     

        # forward the blocks of the transformer
        presents = []
        for block, past in zip(self.transformer.h, past_kvs):
            x, present = block(x, past_kv=past)
            presents.append(present)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, presents

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        hf_token = os.getenv("HF_TOKEN")
        try:
            model_hf = GPT2LMHeadModel.from_pretrained(
                model_type,
                cache_dir=CACHE_DIR,
                local_files_only=True
            )
            print("Loaded model from cache")

        except Exception as e:
            print(f"Error occurred: {e}")
            print("Downloading model...")
            model_hf = GPT2LMHeadModel.from_pretrained(
                model_type,
                token=hf_token,
                cache_dir=CACHE_DIR
            )
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


