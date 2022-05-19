# coding: utf-8
"""
Attention modules
"""

from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as dist

from entmax import Sparsemax, Entmax15, EntmaxBisect
from .basis_functions import (PowerBasisFunctions,
                                     SineBasisFunctions,
                                     CosineBasisFunctions,
                                     GaussianBasisFunctions)
from .continuous_sparsemax import ContinuousSparsemax
from .continuous_softmax import ContinuousSoftmax

import math

import numpy as np

import pickle

import matplotlib.pyplot as plt



class LongTermAttention(nn.Module):
    def __init__(self, head_size:int , length: int, target_len:int,  attn_func: str, attn_num_basis: int,
                n_layers: int, n_heads: int, mask: bool, kl_regularizer: bool, sigma_0, sticky_memories, **kwargs):

        super(LongTermAttention, self).__init__()

        self.device = 'cuda'
        self.length = length #memory length
        self.target_len = target_len #target length / transformer length
        self.head_size = head_size
        self.attn_num_basis = attn_num_basis
        self.attn_func = attn_func # normalizing function
        self.n_head = n_heads
        
        self.kl_regularizer = kl_regularizer
        self.sigma_0 = sigma_0

        self.proj_query = nn.Linear(n_heads*head_size, n_heads*head_size, bias=False)
        self.proj_key = nn.Linear(n_heads*head_size, n_heads*head_size, bias=False)
        self.proj_value = nn.Linear(n_heads*head_size, n_heads*head_size, bias=False)

        self.attn_out = nn.Linear(n_heads*head_size, n_heads*head_size, bias=False)
        
        self.mask=mask

        if self.mask:
            self.mask_net=torch.nn.Conv1d(n_heads*head_size, n_heads*head_size,3,padding=1)
        
        
        self.sticky_memories=sticky_memories
        if self.sticky_memories:
            self.attn_past=None

        self.nb_samples=512 # number of samples used for update
        self.tau = 0.5 #compressing factor
        self.count = 0

        self.x_past=None # previous memory vectors
        self.B_past=None # previous coefficient matrix

        self.ridge_penalty=0.5 # ridge penalty
        padding = True

        self.spacing='linear'

        def compute_G(l, psi, positions, padding=True):

            F = torch.zeros(self.attn_num_basis, positions.size(0))

            basis_functions = psi
            F[:, :] = basis_functions.evaluate(positions.unsqueeze(1)).t()

            I = torch.eye(self.attn_num_basis)
            G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse())

            if padding:
                if l % 2:
                    G = G[((l-1)//2):(-(l-1)//2), :]
                else:
                    G = G[(l//2):-(l//2), :]

            return G.to(self.device)

        
        self.mu = nn.Linear(attn_num_basis, 1, bias=False)
        self.sigma = nn.Linear(attn_num_basis, 1, bias=False)
        self.softplus = torch.nn.Softplus()

        # normalizing function
        if attn_func == 'softmax':
            self.transform = ContinuousSoftmax(psi=None)
        elif attn_func == 'sparsemax':
            self.transform = ContinuousSparsemax(psi=None)
        else:
            assert False

        # get basis functions psi
        sigmas = [.005,.01] # basis function sigmas
        if attn_num_basis % len(sigmas):
            attn_num_basis += (len(sigmas) - attn_num_basis % len(sigmas))

        self.psi=[None]
        self.Gs=[None for _ in range(length+1)]
        self.psi=[None]
        lengths=[]
        for i in range(length):
            self.psi.append([])
            lengths.append(i+1)
        if length not in lengths:
            lengths.append(length)
        for l in lengths:
            # get positions for memory vectors
            self.add_gaussian_basis_functions(self.psi[l], attn_num_basis, sigmas, device=self.device)

            if self.spacing=='linear':
                if padding:
                    if l % 2:
                        shift = 1 / float(l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)
                else:
                    shift = 1 / float(2*l)
                    positions = torch.linspace(shift, 1-shift, l).to(self.device)
            elif self.spacing=='log':
                if padding:
                    if l % 2:
                        shift = 1 / float(l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)

                    pos = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
                    positions = torch.cat([positions[:int(l/2)],pos.to(self.device),positions[-int(l/2):]])

                else:
                    positions = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
        
            # compute basis functions
            self.Gs[l]=compute_G(l, self.psi[l][0], positions, padding=padding) # [L,N]
            self.positions = positions[int(l/2):-int(l/2)]

        # compute samples for memory update
        tm_tau = torch.arange(1,self.nb_samples+1).float()
        tm_l = torch.arange(self.nb_samples+1,length+self.nb_samples+1).float()
        tm_tau = tm_tau*self.tau/self.nb_samples # positions of old vectors
        tm_l = self.tau + (1-self.tau)*(tm_l-self.nb_samples)/length # positions of new vectors
        positions_inf = torch.cat([tm_tau, tm_l],0).to(self.device) # positions

        if padding:
            if l % 2:
                shift = 1 / float(length+self.nb_samples)
                positions_pad_ = torch.linspace(-.5+shift, 0, 2*(length+self.nb_samples)-1).to(self.device)
            else:
                shift = 1 / float(2*length+self.nb_samples)
                positions_pad = torch.linspace(-.5+shift, 1.5-shift, 2*(length+self.nb_samples)).to(self.device)
            positions_pad_ = torch.FloatTensor([i for i in positions_pad if i<0]).to(self.device)
            positions_pad__ = torch.FloatTensor([i for i in positions_pad if i>1]).to(self.device)
            positions_inf = torch.cat([positions_pad_,positions_inf,positions_pad__], dim=0)

        self.samples=None
        for t in tm_tau:
            if self.samples is None:
                self.samples = self.psi[l][0].evaluate(t/self.tau)
            else:
                self.samples = torch.cat([self.samples,self.psi[l][0].evaluate(t/self.tau)], dim=0)

        # compute G for the infinite case
        self.G_inf = compute_G(self.nb_samples+length, self.psi[l][0], positions_inf, padding=padding) #[L+nb_samples,N]

        if self.sticky_memories:
            self.bins = torch.linspace(0,1,129).to(device=self.device)
            self.nb_bins_cat = 1
            self.bins_cat = dist.Categorical(torch.ones(self.nb_bins_cat))


    def add_gaussian_basis_functions(self, psi, nb_basis, sigmas, device):
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas))
        mu = mu.flatten().to(device)
        sigma = sigma.flatten().to(device)
        self.basis_mu=mu
        self.basis_sigma=sigma
        assert mu.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu=mu, sigma=sigma))

    def score(self, query, keys):
        query = query/ (self.d_head ** 0.5) # divide by sqrt(d_head) [B,h,q,d]
        keys = keys.transpose(-1, -2) #[B,h,d,N]
        scores = torch.matmul(query, keys) #[B,h,q,N] 
        return scores

    def value_function(self, x, inf=False):
        if inf:
            G = self.G_inf # [nb_sample+L,N]
        else:
            G = self.Gs[x.size(-1)] # [L,N]

        B = torch.matmul(x, G) # [B,e,N]
        B = B.permute(0,2,1) # [B,N,e]
        
        return B

    def update_inf(self, x):
        if self.B_past is not None:       
            if self.sticky_memories:

                n = dist.Normal(self.attn_past[0],self.attn_past[1])
                
                bins = self.bins.clone()
                bins[0]=-.000001
                bins[-1]=1.000001

                p = (n.cdf(bins[1:].repeat(self.attn_past[0].size(1),x.size(0),1).permute(2,1,0))
                    -n.cdf(bins[:-1].repeat(self.attn_past[0].size(1),x.size(0),1).permute(2,1,0))).sum(-1).transpose(1,0)

                p = (p/p.sum(-1).repeat(p.size(-1),1).transpose(1,0)) 
            
                p = torch.clamp(p, min=0)
                p = dist.Categorical(p)


                b = p.sample((self.nb_samples,))
                
                t = self.bins_cat.sample((self.nb_samples,self.attn_past[0].size(0))).to(device=self.device)

                ts = (t*(self.bins[b+1]-self.bins[b])/self.nb_bins_cat +self.bins[b]).transpose(1,0)

                ts = torch.sort(ts,-1)[0]
            
                samples=torch.zeros(x.size(0),self.nb_samples,self.attn_num_basis).to(device=self.device)
                for i in range(len(ts)):
                    samples[i] = self.psi[self.length][0].batch_evaluate(ts[i])

                xm_tau = self.B_past.transpose(-1,-2).matmul(samples.transpose(-1,-2)) # [B,e,nb_samples]
            
            else:
                xm_tau = self.B_past.transpose(-1,-2).matmul(self.samples.transpose(-1,-2)) # [B,e,nb_samples]
            
            x = torch.cat([xm_tau,x], dim=2) # [B,e,nb_samples+L]
            B = self.value_function(x, inf=True) # [B,N,e]
        else:
            B = self.value_function(x)
        
        self.B_past=B.detach()
        self.x_past=x
        return B


    def forward(self, k, q, new_doc, generating=False, layer_n=None):
       
        if not generating:
            k=k[:,-512:,:]

        batch_size = 1 # k.size(0) #batch size
        qlen = q.size(2) #query length
        if not generating:
            self.klen = k.size(1) #key length
        self.d_head = self.head_size #head size

        # clean memory if going through different document
        if new_doc:
            self.B_past=None 
            self.x_past=None
            self.count=0

        if not generating:
            k = k.permute(0,2,1) # [B,e,L]
            if self.mask:
                reg_mask=torch.sigmoid(self.mask_net(k))
                k = k*reg_mask
            elif self.mask:
                k = k*reg_mask

        # perform memory update
        if not generating:
            B = self.update_inf(k)
            self.count+=self.klen
        else:
            B=self.B_past
        
        keys = self.proj_key(B)
        values = self.proj_value(B)

        query = q
        keys = keys.view(batch_size,self.attn_num_basis,self.n_head,self.d_head).transpose(1,2) # [B,h,N,d]
        values = values.view(batch_size,self.attn_num_basis,self.n_head,self.d_head).transpose(1,2) # [B,h,N,d]
        
        #compute scores
        scores = self.score(query, keys) #[B,h,q,N] 

        #compute mu and sigma
        mu = torch.sigmoid(self.mu(scores)) #[B,h,q] 
        sigma_sq = self.softplus(self.sigma(scores))#[B,h,q] 

        mu = mu.view(-1)
        sigma_sq = torch.clamp(sigma_sq, min=1e-4).view(-1)

        if self.sticky_memories:
            self.attn_past=[mu.view(batch_size,-1),sigma_sq.view(batch_size,-1)**(1/2)]


        if self.kl_regularizer:
            sigma_0_sq = self.sigma_0**2
            kl_reg = 1/2 * ( sigma_sq.view(batch_size,-1) / sigma_0_sq - 
                            torch.log(sigma_sq.view(batch_size,-1)/sigma_0_sq) -1 )
            

        # pass parameters to theta
        theta = torch.zeros(batch_size*self.n_head*qlen, 2, device=self.device)  # [B*h*q, 2]
        theta[:, 0] = mu / sigma_sq
        theta[:, 1] = -1. / (2. * sigma_sq)

        # get basis functions
        self.transform.psi = self.psi[self.klen]

        #compute basis functions expectation
        r = self.transform(theta) # [B*h*q,N] 

        r = r.view(batch_size,self.n_head,qlen,self.attn_num_basis).permute(0,1,3,2) # [B,h,N,q]

        values = values.transpose(-1,-2) # [B,h,d,N]
        
        context = torch.matmul(values,r) # [B,h,d,q]

        context = context.permute(0,3,1,2) # [q,B,h,d]
        context = context.contiguous().view(batch_size,qlen,self.n_head*self.d_head) # [q,B,e]

        context = self.attn_out(context)

        if self.kl_regularizer:
            return context, kl_reg
        else:
            return context
        
    @property
    def _query_dim(self):
        return self.query_layer.in_features

    def __repr__(self):
        return "ContinuousAttention"

