# used to fit dat with static stress tensor

import numpy as np
import torch 
import opt_einsum as oe
from torch.autograd.functional import jacobian
#from src.MODEL import *
#============================calculate the energy===================================
class Property(torch.nn.Module):
    def __init__(self,getdensity):            ##delete nnmodlist
        super(Property,self).__init__()
        self.getdensity=getdensity

    def forward(self,cart,numatoms,species,atom_index,shifts,volume,force_sizee,create_graph=True):
        cart.requires_grad=True
        species=species.view(-1)     
        dist_vec,selected_id_config,output = self.getdensity(cart,numatoms,species,atom_index,shifts)
        #output=self.nnmod(density,species).view(numatoms.shape[0],-1)
        varene=torch.sum(output,dim=1)
        #stress
        grad_outputs=torch.ones(numatoms.shape[0],device=cart.device)
        grad_dist_vec=torch.autograd.grad(varene,dist_vec,grad_outputs=grad_outputs,\
        create_graph=create_graph,only_inputs=True,allow_unused=True)[0]

        stress_tmp=torch.einsum("ij,ik->ijk",grad_dist_vec,dist_vec)
        stress=torch.zeros(numatoms.shape[0],3,3,device=cart.device)
        stress.index_add_(0,selected_id_config,stress_tmp)
        stress=-(torch.einsum('ijk,i->ijk',stress,1/volume)).view(numatoms.shape[0],-1)
        #force
        force=torch.zeros(numatoms.shape[0],force_sizee[0],device=cart.device)
        force0=-torch.autograd.grad(dist_vec,cart,grad_outputs=grad_dist_vec,\
        create_graph=create_graph,only_inputs=True,allow_unused=True)[0].view(numatoms.shape[0],-1)
        force[:,:force0.size(1)]=force0
        return varene,stress,force

