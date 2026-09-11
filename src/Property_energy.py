import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
#from src.MODEL import *
#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,getdensity):
        super(Property,self).__init__()
        self.getdensity=getdensity

    def forward(self,cart,numatoms,species,atom_index,shifts,volume,force_sizee,create_graph=None):
        cart.requires_grad=True
        species=species.view(-1)
        dist_vec,selected_id_config,output = self.getdensity(cart,numatoms,species,atom_index,shifts)
        varene=torch.sum(output,dim=1)
        return varene,

