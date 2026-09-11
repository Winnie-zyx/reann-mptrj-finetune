import torch
from collections import OrderedDict
from torch import nn
from torch.nn import Linear,Dropout,BatchNorm1d,Sequential,LayerNorm
from torch.nn import Softplus,GELU,Tanh,SiLU
from torch.nn.init import xavier_uniform_,zeros_,constant_
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, nl, dropout_p, actfun, table_norm=True):
        super(ResBlock, self).__init__()
        # activation function used for the nn module
        nhid=len(nl)-1
        sumdrop=np.sum(dropout_p)
        modules=[]
        for i in range(1,nhid):
            modules.append(actfun(nl[i-1],nl[i]))
            if table_norm: modules.append(LayerNorm(nl[i]))
            if sumdrop>=0.0001: modules.append(Dropout(p=dropout_p[i-1]))
            #bias = not(i==nhid-1)
            linear=Linear(nl[i],nl[i+1])
            if i==nhid-1: 
                zeros_(linear.weight)
            else:
                xavier_uniform_(linear.weight)
            if i==nhid-1: zeros_(linear.bias)
            modules.append(linear)
        self.resblock=Sequential(*modules)

    def forward(self, x):
        return self.resblock(x) + x

#==================for get the atomic energy=======================================
class NNMod(torch.nn.Module):
   def __init__(self,outputneuron,nblock,nl,dropout_p,actfun,initbias=torch.zeros(1),initpot=0,table_norm=True,layernorm=True):          ##delete maxnumtype and atomtype
      """
      nl: is the neural network structure;
      outputneuron: the number of output neuron of neural network
      """
      super(NNMod,self).__init__()
      self.register_buffer("initpot",torch.Tensor([initpot]))
      # create the structure of the nn     
      self.outputneuron=outputneuron
      #elemental_nets=OrderedDict()
      sumdrop=np.sum(dropout_p)
      with torch.no_grad():             ##delete torch.no_grad, nl.append(nl[1]), nhid=len(nl)-1
          if nblock>1.5:
              if abs(nl[1]-nl[-1])>0.5: nl.append(nl[1])
          nhid=len(nl)-1
          modules=[]
          linear=Linear(nl[0],nl[1])
          xavier_uniform_(linear.weight)
          modules.append(linear)
          if nblock > 1.5:
              for iblock in range(nblock):
                  modules.append( * [ResBlock(nl,dropout_p,actfun,table_norm=table_norm)])
          else:
              for ilayer in range(1,nhid):
                  modules.append(actfun(nl[ilayer-1],nl[ilayer]))
                  if layernorm: modules.append(LayerNorm(nl[ilayer]))
                  if sumdrop>=0.0001: modules.append(Dropout(p=dropout_p[ilayer-1]))
                  linear=Linear(nl[ilayer],nl[ilayer+1])
                  xavier_uniform_(linear.weight)
                  modules.append(linear)
          modules.append(actfun(nl[nhid-1],nl[nhid]))
          linear=Linear(nl[nhid],self.outputneuron)
          zeros_(linear.weight)
          linear.bias[:]=initbias[:]
          modules.append(linear)
      self.net= Sequential(*modules)    ##按顺序保存module，之后索引的是module，是一个列表封装成的字典
   def forward(self, density):
       # Process all atoms with the same neural network
       #output = self.net(density)
       return self.net(density)
