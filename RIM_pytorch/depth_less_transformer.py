from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Linear, Parameter, ParameterList
from torch.func import vmap, functional_call

from torch_einops_utils import pack_with_inverse
from PoPE_pytorch import PoPE, flash_attn_with_pope

from einops import einsum, repeat, rearrange, pack
from einops.layers.torch import Rearrange, Reduce

# einstein notation

# m - messages
# l - bLocks
# b - batch
# n - sequence
# d - feature dimension
# i, j - source and target sequence for attention
# h - attention heads

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# init

@torch.no_grad()
def init_ensemble_weights_(params, names):
    for name, param in zip(names, params):
        if 'norm' in name:
            nn.init.ones_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
        elif 'weight' in name:
            fan_in = param.shape[-1]
            bound = fan_in ** -0.5
            nn.init.uniform_(param, -bound, bound)

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = False,
        key_rmsnorm = False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.causal = causal

        self.norm = nn.RMSNorm(dim)
        self.maybe_key_norm = nn.RMSNorm(dim_head) if key_rmsnorm else nn.Identity()

        self.to_q = LinearNoBias(dim, dim_inner)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)

        self.to_out = LinearNoBias(dim_inner, dim)

        self.to_gates = nn.Sequential(LinearNoBias(dim, heads), Rearrange('... n h -> ... h n 1'))

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens,
        context = None,
        pos_emb = None
    ):
        device = tokens.device
        tokens, inverse_pack = pack_with_inverse(tokens, '* n d')
        tokens = self.norm(tokens)

        if exists(context):
            context, _ = pack_with_inverse(context, '* n d')
        else:
            context = tokens

        q, k, v = (self.to_q(tokens), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = map(self.split_heads, (q, k, v))

        k = self.maybe_key_norm(k)

        if exists(pos_emb):
            out = flash_attn_with_pope(
                q, k, v,
                pos_emb = pos_emb,
                causal = self.causal
            )
        else:
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if self.causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim = -1)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.to_gates(tokens).sigmoid() * out

        out = self.merge_heads(out)
        out = self.to_out(out)
        return inverse_pack(out)

# swiglu ff - Shazeer et al

class Feedforward(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor * 2 / 3)

        self.norm = nn.RMSNorm(dim)
        self.keys = Linear(dim, dim_inner * 2)
        self.values = Linear(dim_inner, dim)

    def forward(
        self,
        tokens
    ):
        queries = self.norm(tokens)
        sim, gates = self.keys(queries).chunk(2, dim = -1)
        sim = sim * F.gelu(gates)
        return self.values(sim)

# ensemble

class Ensemble(Module):
    def __init__(
        self,
        net: Module,
        ensemble_size: int
    ):
        super().__init__()
        repeat_blocks = Reduce('... -> l ...', 'repeat', l = ensemble_size)

        named_params = dict(net.named_parameters())

        # avoid the issue with period in the parameter names

        self.param_names = named_params.keys()
        self.net_parameters = ParameterList([Parameter(repeat_blocks(param).clone()) for param in named_params.values()])

        init_ensemble_weights_(self.net_parameters, self.param_names)

        # vmapping

        def net_forward(params, tokens, *args, **kwargs):
            return functional_call(net, params, args = (tokens, *args), kwargs = kwargs)

        self.net_forward = vmap(net_forward, in_dims = 0)

    def forward(self, tokens, *args, **kwargs):
        params = dict(zip(self.param_names, self.net_parameters))
        return self.net_forward(params, tokens, *args, **kwargs)

# classes

class DepthlessTransformer(Module):
    def __init__(
        self,
        dim,
        num_blocks = 6,
        num_message_exchanges = 6,
        dim_head = 64,
        heads = 8,
        causal = False,
        ff_expansion_factor = 4.,
        num_tokens = None,
        use_pope = False,
    ):
        super().__init__()

        self.num_message_exchanges = num_message_exchanges

        self.num_blocks = num_blocks
        repeat_blocks = Reduce('... -> l ...', 'repeat', l = num_blocks)
        self.repeat_blocks = repeat_blocks

        self.use_pope = use_pope
        if use_pope:
            self.pope = PoPE(dim = dim_head, heads = heads)

        # define attention and feedforward

        attn = Attention(dim, causal = causal, dim_head = dim_head, heads = heads)
        ff = Feedforward(dim, ff_expansion_factor)

        # the attention residual, or just putting together the information coming from various recurrent modules

        self.attn_residual = Attention(dim, key_rmsnorm = True, dim_head = dim_head, heads = heads)

        # make ensemble

        self.attn_ensemble = Ensemble(attn, num_blocks)
        self.ff_ensemble = Ensemble(ff, num_blocks)

        # readout

        self.query_readout = nn.Parameter(torch.randn(dim) * 1e-2)
        self.readout = nn.Sequential(nn.RMSNorm(dim), LinearNoBias(dim, num_tokens)) if exists(num_tokens) else None

    def forward(
        self,
        tokens,
        return_messages = False
    ):
        batch, seq_len, blocks = *tokens.shape[:2], self.num_blocks

        tokens = self.repeat_blocks(tokens) # (l b n d)

        # reframed as recurrent processing of tokens with message passing (attention residual)

        messages = [tokens]

        attn_kwargs = dict()
        if self.use_pope:
            pos_emb = self.pope(seq_len)
            attn_kwargs = dict(pos_emb = pos_emb)

        # message passing

        for count in enumerate(range(self.num_message_exchanges), start = 1):
            is_last = count == self.num_message_exchanges

            # representations go into all of the blocks at once, without any notion of depth

            attended = self.attn_ensemble(tokens, **attn_kwargs)
            retrieved_memories = self.ff_ensemble(tokens)

            # add outputs to processed messages

            messages.extend([attended, retrieved_memories])

            # on the last round, one query token from readout block aggregate / attention residual

            if is_last:
                continue

            # then we just do attention pooling (attention 'residual') for next round
            # will use the initial messages coming in as the queries, all products of all the blocks become messages - voting phase

            packed_messages, _ = pack(messages, '* b n d') # (message blocks) packed
            all_messages = repeat(packed_messages, 'm b n d -> (b l) n m d', l = blocks)

            message_queries, inverse_pack_blocks = pack_with_inverse(tokens, '* n d') # (m b n d)
            message_queries = rearrange(message_queries, '... n d -> ... n 1 d')

            # each message producer attends to all messages (and their history) by all other producers

            pooled_messages = self.attn_residual(message_queries, all_messages)

            pooled_messages = inverse_pack_blocks(pooled_messages, '* n one d')

            pooled_messages = rearrange(pooled_messages, 'l b n 1 d -> l b n d')

            # keep iterating

            tokens = pooled_messages

        # the readout itself is just another message producer

        queries = repeat(self.query_readout, 'd -> b n 1 d', b = batch, n = seq_len)

        all_messages = rearrange(messages, 'm l b n d -> b n (m l) d')

        readout_input = self.attn_residual(queries, all_messages)

        readout_input = rearrange(readout_input, 'b n 1 d -> b n d')

        if not exists(self.readout):
            return readout_input

        logits = self.readout(readout_input)

        if not return_messages:
            return logits

        return logits, messages
