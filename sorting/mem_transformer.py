import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ContinuousAttention
from long_term_attention import LongTermAttention 
from long_term_attention_transformer import LongTermAttentionTransformer


from functools import partial

sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, kl_regularizer, dropatt=0, 
                 tgt_len=None, eval_tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 long_term_attention=False, long_term_attention_basis=None, long_term_attention_norm=None,
                 infinite_memory=False, n_layers=None, augment_len=None, affines=False, mask=False, mask_type=None,
                 share_mask=False, sigma_0=None, mu_0=None):
        super(RelMultiHeadAttn, self).__init__()

        self.use_long_term_attention = long_term_attention
        self.long_term_attention_basis = long_term_attention_basis
        self.long_term_attention_norm = long_term_attention_norm
        self.infinite_memory = infinite_memory
        self.n_layers = n_layers

        self.augment_len = augment_len
        self.affines = affines
        self.mask = mask
        self.mask_type = mask_type
        self.kl_regularizer=kl_regularizer
        self.sigma_0=sigma_0
        self.mu_0=mu_0

        self.mem_len = mem_len
        self.tgt_len = tgt_len
        self.eval_tgt_len = eval_tgt_len
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.qkv_net_ = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.attn_drop = dropatt
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        
        long_mem_len=self.augment_len
        
        if self.use_long_term_attention:    
            long_term_attn_mechanism = partial(LongTermAttention,
                                            attn_num_basis=self.long_term_attention_basis,
                                            head_size=self.d_head,
                                            length=long_mem_len,
                                            target_len=self.tgt_len,
                                            attn_func=self.long_term_attention_norm,
                                            infinite_memory=self.infinite_memory,
                                            n_layers=self.n_layers,
                                            attn_drop=self.attn_drop,
                                            n_heads=self.n_head,
                                            d_model=self.d_model,
                                            affines=self.affines,
                                            mask=self.mask,
                                            mask_type=self.mask_type,
                                            kl_regularizer=self.kl_regularizer,
                                            sigma_0=self.sigma_0,
                                            mu_0=self.mu_0,
                                            )
                                        
            self.long_term_attention=long_term_attn_mechanism()
            self.aux_long_term = False
            self.a_long_term=None
            self.new_doc=True

        if self.kl_regularizer:
            self.kl_reg=None


    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, layer_n=None, a_long_term=None, reg_mask=None,
                doc_final=False):
        a_long_term=None

        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        #if mems is not None:
        #    print(mems.shape)

        if self.kl_regularizer:
            self.kl_reg=None

        if mems is not None and len(mems)==0:
            self.new_doc=True
            
        if mems is not None and len(mems)>0 and self.use_long_term_attention:

            cat = torch.cat([mems[-self.mem_len:], w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]

            klen = w_head_k.size(0)

            k = mems[:-self.mem_len]
            q = w_head_q

            if len(k)>0:
                if self.kl_regularizer:
                    a_long_term, self.kl_reg = self.long_term_attention(k, q, new_doc=self.new_doc, layer_n=layer_n,
                                                        reg_mask=reg_mask, doc_final=doc_final)
                else:
                    a_long_term = self.long_term_attention(k, q, new_doc=self.new_doc, layer_n=layer_n,
                                                        reg_mask=reg_mask, doc_final=doc_final)
                self.new_doc=False      



        elif mems is not None and not self.use_long_term_attention:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]

        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        self.past_x = True

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        r_head_k = r_head_k.view(r_head_k.size(0), self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)


        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)

        # combining long term context vector with context vector
        if self.use_long_term_attention and a_long_term is not None:    
            attn_out += a_long_term
        

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        if self.kl_regularizer:
            return output, a_long_term, self.kl_reg
        return output, a_long_term

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, kl_regularizer, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,d_head, dropout, kl_regularizer, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

        self.kl_regularizer=kl_regularizer

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, layer_n=None, a_long_term=None,
                    reg_mask=None ,doc_final=False):

        if self.kl_regularizer:
            output, a_long_term, kl_reg = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,attn_mask=dec_attn_mask,mems=mems,
                             layer_n=layer_n, a_long_term=a_long_term, reg_mask=reg_mask, doc_final=doc_final)
        else:
            output, a_long_term = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,attn_mask=dec_attn_mask,mems=mems,
                             layer_n=layer_n, a_long_term=a_long_term, reg_mask=reg_mask, doc_final=doc_final)
        output = self.pos_ff(output)

        if self.kl_regularizer:
            return output, a_long_term, kl_reg    
        return output, a_long_term


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None, 
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, eval_tgt_len=None, ext_len=None, mem_len=None, 
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1, 
                 sample_softmax=-1,
                 long_term_attention=False, long_term_attention_basis=None,
                 long_term_attention_norm=None, infinite_memory=False, n_layers=None, 
                 augment_len=None, affines=False,mask=False,
                 mask_type=None,kl_regularizer=False, sigma_0=None, mu_0=None):

        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.eval_tgt_len = eval_tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()

        self.use_long_term_attention = long_term_attention

        self.augment_len=augment_len

        self.mask=mask
        self.mask_type=mask_type
        self.kl_regularizer = kl_regularizer
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0


        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout, 
                    long_term_attention=long_term_attention,
                    long_term_attention_basis=long_term_attention_basis,
                    long_term_attention_norm=long_term_attention_norm,
                    infinite_memory=infinite_memory, n_layers=n_layer,
                    tgt_len=tgt_len, eval_tgt_len=eval_tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, pre_lnorm=pre_lnorm, augment_len=augment_len,
                    mask=mask, mask_type=mask_type,
                    kl_regularizer=kl_regularizer, sigma_0=sigma_0, mu_0=mu_0,
                    ))

        self.sample_softmax = sample_softmax

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=21)

        self.proj_output=nn.Linear(d_model,n_token, bias=False)

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
         # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        if self.augment:
            mem_len=self.mem_len+self.augment_len
        else:
            mem_len=self.mem_len

        with torch.no_grad():
            new_mems = []
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[-mem_len:].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, doc_final=False):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        if self.use_long_term_attention:
            mlen = mems[0].size(0) if mems is not None else 0
            if mlen>self.mem_len:
                mlen=self.mem_len
        else:
            mlen = mems[0].size(0) if mems is not None else 0

        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]
        
        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        if self.kl_regularizer:
            kl_regs=None

        hids.append(core_out)
        a_long_term=None

        
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            reg_mask = None

            if self.kl_regularizer:
                core_out, a_long_term, kl_reg = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask,
                        mems=mems_i, layer_n=i, a_long_term=a_long_term, reg_mask=reg_mask, doc_final=doc_final)
            else:
                core_out, a_long_term = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask,
                        mems=mems_i, layer_n=i, a_long_term=a_long_term, reg_mask=reg_mask, doc_final=doc_final)
            hids.append(core_out)

            if self.kl_regularizer:
                if kl_regs is None:
                    kl_regs=kl_reg
                else:
                    kl_regs+=kl_reg

        core_out = self.drop(core_out)

        if doc_final:
            core_out = self.proj_output(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        if self.kl_regularizer:
            return core_out, new_mems, kl_regs    
        return core_out, new_mems

    def forward(self, data, target, *mems, doc_final=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems: mems = self.init_mems()
        tgt_len = target.size(0)

        if self.kl_regularizer:
            hidden, new_mems, kl_reg = self._forward(data, mems=mems, doc_final=doc_final)
        else:
            hidden, new_mems = self._forward(data, mems=mems, doc_final=doc_final)

        pred_hid = hidden[-tgt_len:]

        
        loss = self.loss(pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1))

        if new_mems is None:
            return [loss], pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1)
        elif self.kl_regularizer:
            return [loss] + new_mems, pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1), kl_reg
        else:
            return [loss] + new_mems, pred_hid.view(-1, pred_hid.size(-1)), target.contiguous().view(-1)

