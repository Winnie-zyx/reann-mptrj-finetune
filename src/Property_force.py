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

    def forward(self,cart,numatoms,species,atom_index,shifts,volume,force_sizee,create_graph=True):
        cart.requires_grad=True
        species=species.view(-1)
        dist_vec,selected_id_config,output = self.getdensity(cart,numatoms,species,atom_index,shifts)
        #density = self.density(cart,numatoms,species,atom_index,shifts)
        #output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        
        force=torch.zeros(numatoms.shape[0],force_sizee[0],device=cart.device)
        grad_outputs=torch.ones(numatoms.shape[0],device=cart.device)
        force0=-torch.autograd.grad(varene,cart,grad_outputs=grad_outputs,\
        create_graph=create_graph,only_inputs=True,allow_unused=True)[0].view(numatoms.shape[0],-1)
        force[:,:force0.size(1)]=force0
        return varene,force

