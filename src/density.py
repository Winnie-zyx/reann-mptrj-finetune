import torch
from torch import nn
from torch import Tensor
from collections import OrderedDict
import numpy as np
import opt_einsum as oe
from src.MODEL import *
from src.activate import Relu_like,Tanh_like 
import gc
from sys import getrefcount
#import MODEL as MODEL

class GetDensity(torch.nn.Module):
    def __init__(self,neigh_atoms, nipsin=2, cutoff=4 ,norbit=64, nwave=8, emb_nblock=1, emb_nl=[1,8], emb_layernorm=True,oc_loop=3, oc_nblock=1, oc_nl=[64,64], oc_dropout_p=[0.0,0.0],oc_layernorm=True, nblock=1, nl=[64,64], dropout_p=[0.0,0.0],layernorm=True, initpot=0.0, Dtype=torch.float32):   ##add atom_species, delete rs and inta, atom_species from inputand transport to torch,tensor([[8.],[1.]])
        super(GetDensity,self).__init__()
        '''
        rs: tensor[nwave] float
        inta: tensor[nwave] float
        nipsin: np.array/list   int
        cutoff: float
        norbit is similar to ncontract, which define in read.py
        '''
        self.nwave=nwave
        self.norbit=norbit 
        self.register_buffer('cutoff', torch.Tensor([cutoff]))          #注册缓冲区
        self.register_buffer('nipsin', torch.tensor([nipsin]))
        self.register_buffer('initpot',torch.Tensor([initpot]))
 
        self.contracted_coeff=nn.parameter.Parameter(torch.nn.init.xavier_uniform_(torch.randn(oc_loop+1,nipsin,nwave,norbit)))     #len(ocmod_list),nipsin+1,nwave,ncontract

        # calculate the angular part 
        npara=[1]
        index_para=torch.tensor([0],dtype=torch.long)
        for i in range(1,nipsin):
            npara.append(np.power(3,i))
            index_para=torch.cat((index_para,torch.ones((npara[i]),dtype=torch.long)*i))
        
        self.register_buffer('index_para',index_para)
        #self.register_buffer('index_para',torch.Tensor([0.,1.,1.,1.,2.,2.,2.,2.,2.,2.,2.,2.,2.]))
        ##nipsin=2: index_para=[0,1,1,1,2,2,2,2,2,2,2,2,2]
        
        alpha=torch.ones(nwave,dtype=Dtype)
        rs=(torch.rand(nwave)*np.sqrt(cutoff)).to(Dtype)
        initbias=torch.randn(nwave,dtype=Dtype)/neigh_atoms
        initbias=torch.hstack((initbias,alpha,rs))
        ##rs and inta are given in read.py

        #embedded nn
        self.emb_neighnn=NNMod(nwave*3,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)          ##emb_nl should be given in input
        initbias=torch.randn(norbit,dtype=Dtype)/neigh_atoms
        self.emb_centernn=NNMod(norbit,emb_nblock,emb_nl,np.array([0]),Relu_like,initbias=initbias,layernorm=emb_layernorm)     #维度怎么给进去？

        ocmod=OrderedDict()
        for i in range(oc_loop):
            initbias=torch.randn(nwave,dtype=Dtype)
            f_oc="memssage_"+str(i)
            ocmod[f_oc]=NNMod(nwave,oc_nblock,oc_nl,oc_dropout_p,Relu_like,initbias=initbias,layernorm=oc_layernorm)
        self.ocmod= torch.nn.ModuleDict(ocmod)
        self.outnn=NNMod(1,nblock,nl,dropout_p,Relu_like,initbias=torch.tensor(np.array([1])),layernorm=layernorm)           #initpot is loaded from dataloader,remember load it in train.py

    #def gaussian(self,distances):
        # Tensor: rs[nwave],inta[nwave] 
        # Tensor: distances[neighbour*numatom*nbatch,1]
        # return: radial[neighbour*numatom*nbatch,nwave]
        #distances=distances.view(-1,1)
        #radial=torch.empty((distances.shape[0],self.rs.shape[0]),dtype=distances.dtype,device=distances.device)              #the shape of rs
        #radial=torch.exp(self.inta * torch.square(distances - self.rs))
        #for itype in range(self.rs.shape[0]):
        #    mask = (species_ == itype)             #待修改
        #    ele_index = torch.nonzero(mask).view(-1)
        #    if ele_index.shape[0]>0:
        #        part_radial=torch.exp(self.inta[itype:itype+1]*torch.square \
        #        (distances.index_select(0,ele_index)-self.rs[itype:itype+1]))
        #        radial.masked_scatter_(mask.view(-1,1),part_radial)
        #return radial
    
    def cutoff_cosine(self,distances):
        # assuming all elements in distances are smaller than cutoff
        # return cutoff_cosine[neighbour*numatom*nbatch]
        return torch.square(0.5 * torch.cos(distances * (np.pi / self.cutoff)) + 0.5)
    
    def angular(self,dist_vec,f_cut):
        # Tensor: dist_vec[neighbour*numatom*nbatch,3]
        # return: angular[neighbour*numatom*nbatch,npara[0]+npara[1]+...+npara[ipsin]]
        totneighbour=dist_vec.shape[0]
        dist_vec=dist_vec.permute(1,0).contiguous()
        orbital=f_cut.view(1,-1)
        angular=torch.empty(self.index_para.shape[0],totneighbour,dtype=f_cut.dtype,device=f_cut.device)
        angular[0]=f_cut
        num=1
        for ipsin in range(1,self.nipsin[0]):
            #orbital=oe.contract("ji,ki -> jki",orbital,dist_vec,backend="torch").reshape(-1,totneighbour)
            orbital=torch.einsum("ji,ki -> jki",orbital,dist_vec).view(-1,totneighbour)
            angular[num:num+orbital.shape[0]]=orbital
            num+=orbital.shape[0]
        return angular  
    
    def forward(self,cart,numatoms,species,atom_index,shifts):
        """
        # input cart: coordinates (nbatch*numatom,3)
        # input shifts: coordinates shift values (unit cell)
        # input numatoms: number of atoms for each configuration
        # atom_index: neighbour list indice
        # species: indice for element of each atom
        #center_factor 
        #neigh_factor
        """
        #print('density atom_index',atom_index.size(),flush=True)
        tmp_index=torch.arange(numatoms.shape[0],device=cart.device)*cart.shape[1]
        self_mol_index=tmp_index.view(-1,1).expand(-1,atom_index.shape[2]).reshape(1,-1)
        #print('density cart',cart.size(),'\n',cart,flush=True)
        id_config=torch.arange(atom_index.shape[1],device=cart.device).reshape(-1,1).repeat(1,atom_index.shape[2]).reshape(-1)        ##atom_index:[2,batchsize,?]

        cart_=cart.flatten(0,1)    ##cart:[batchsize,numatoms,3], cart_:[batchsize*numatoms,3]
        totnatom=cart_.shape[0]    ##totnatom:[batchsize*numatoms]
        padding_mask=torch.nonzero((shifts.view(-1,3)>-1e3).all(1)).view(-1)
        # get the index for the distance less than cutoff (the dimension is reduntant)
        atom_index12=(atom_index.view(2,-1)+self_mol_index)[:,padding_mask]   ##atom_index12:[2,neigh],neigh is the total number of pairs in cutoff, only truely pairs.
        #print(atom_index12)
        selected_id_config=id_config[padding_mask]
        #print('density selected_id_config',selected_id_config.size(),'\n',selected_id_config,flush=True)
        selected_cart = cart_.index_select(0, atom_index12.view(-1)).view(2, -1, 3)  ##selected_cart:[2,neigh,3]
        shift_values=shifts.view(-1,3).index_select(0,padding_mask)
        dist_vec = selected_cart[0] - selected_cart[1] + shift_values  ##dist_vec:[neigh,3]
        distances = torch.linalg.norm(dist_vec,dim=-1)   ##distances:[neigh]
        #dist_vec=dist_vec/distances.view(-1,1)
        #species_1=species.reshape(-1,1)    ##species:[batchsize*numatoms],species_1:[batchsize*numatoms,1]
        species_1=species.add(1)
        #species_1=species_1.float()        
        #print(species)
        #print(species_1)
        num_classes = 118
        one_hot_encodings = torch.nn.functional.one_hot(species_1, num_classes)  ##one_hot_encodings:[batchsize*numatoms,118]
        one_hot_encodings = one_hot_encodings.float()
        #print(one_hot_encodings.size())
        #print(one_hot_encodings)
        #print(species_1)
        center_coeff=self.emb_centernn(one_hot_encodings)
        #center_coeff=self.emb_centernn(species_1)   ##center_coeff:[batchsize*numatoms,norbit]
        expand_spec=torch.index_select(one_hot_encodings,0,atom_index12.view(-1)).view(2,-1,118)   ##expand_spec:[2,neigh,118]
        #expand_spec=torch.index_select(species_1,0,atom_index12.view(-1)).view(2,-1,1)   ##expand_spec:[2,neigh,1]
        #hyper_spec=expand_spec[0]*expand_spec[1]/(expand_spec[0]+expand_spec[1])  #why dispose the species of center and neigh atoms by this way
        hyper_spec=expand_spec[0]+expand_spec[1]  #why dispose the species of center and neigh atoms by this way
        #hyper_spec=expand_spec[0]
        #print(species)
        #print(hyper_spec)
        #print("***")
        neigh_emb=self.emb_neighnn(hyper_spec).T.contiguous()    ##[3*nwave,neigh]
        #print(neigh_emb)
        #species_ = species.index_select(0,atom_index12[1])          #delete
        cut_distances=self.cutoff_cosine(distances)
        radial_func=torch.exp(-torch.square(neigh_emb[self.nwave:self.nwave*2]*(distances-neigh_emb[self.nwave*2:])))    ##radial_func:[nwave,neigh]
        nangular=self.angular(dist_vec,cut_distances)   ##[lenth,neigh]  lenth=npara[0]+npara[1]+...+npara[ipsin]
        orbital=torch.einsum("ji,ki -> ijk",nangular,radial_func).contiguous()
        #orbital = oe.contract("ji,ki -> ijk",nangular,radial_func,backend="torch")    ##orbital:[neigh,lenth,nwave]
        weight_orbital=torch.einsum("ijk,ki->ijk",orbital,neigh_emb[:self.nwave]).contiguous()     #weight_orbital:[neigh,lenth,nwave]
        zero_orbital=cart.new_zeros((cart_.shape[0],nangular.shape[0],self.nwave),dtype=cart.dtype,device=cart.device)     #nangular.shape[0]=1+3+9=13, zero_orbital:[batchsize*numatoms,lenth,nwave]
        contracted_coeff=torch.index_select(self.contracted_coeff,1,self.index_para)
        center_orbital=torch.index_add(zero_orbital,0,atom_index12[0],weight_orbital)   #atom_index12:[2,batch*numatoms],center_orbital:[batch*numatoms,lenth,nwave]
        #contracted_coeff== orb_coeff
        contracted_orbital=torch.einsum("ijk,jkm->ijm",center_orbital,contracted_coeff[0])
        ## center_orbital:[batch*numatoms,lenth,nwave], contracted_coeff[0]:[lenth,nwave,norbit]
        density=torch.einsum("ijm,ijm ->im",contracted_orbital,contracted_orbital)+center_coeff
        #density=self.obtain_orb_coeff(0,totnatom,orbital,atom_index12,orb_coeff).view(totnatom,-1)
        iter_coeff=neigh_emb[:self.nwave].T.contiguous()    ##iter_coeff:[neigh,nwave]
        for ioc_loop, (_, m) in enumerate(self.ocmod.items()):
            nnout=m(density)
            iter_coeff = iter_coeff + torch.index_select(nnout,0,atom_index12[1])
            density,center_orbital=self.density0(orbital,cut_distances,iter_coeff,atom_index12[0],atom_index12[1],contracted_coeff[ioc_loop+1],zero_orbital,center_orbital,center_coeff)
            #density = self.obtain_orb_coeff(ioc_loop+1,totnatom,orbital,atom_index12,orb_coeff)
#----------------output calculation------------------------------------
        mask_species = (species>-0.5)
        output1= self.outnn(density)    ##[numatoms,norbit]
        out = torch.einsum("ij,i->ij",output1,mask_species)
        output=out.view(numatoms.shape[0],-1)
        #print(density)
        #del selected_cart,species_1,center_coeff,expand_spec,hyper_spec,neigh_emb,cut_distances,radial_func,nangular,orbital,weight_orbital,zero_orbital,contracted_coeff,center_orbital,contracted_orbital,iter_coeff
        #gc.collect()
        #torch.cuda.empty_cache()
        #print(f"{center_coeff}: size={center_coeff.size()}, allocated memory={torch.cuda.memory_allocated()/1000000000} bytes")
        #log_tensor_memory(selected_cart, "selected_cart") 
        #print(getrefcount(selected_id_config))
        #return density,dist_vec,selected_id_config,output
        return dist_vec,selected_id_config,output
 
    def density0(self,orbital,cut_distances,iter_coeff,index_center,index_neigh,contracted_coeff,zero_orbital,center_orbital,center_coeff):
        weight_orbital = torch.einsum("ik,ijk -> ijk",iter_coeff,orbital)+torch.einsum("ijk,i->ijk",torch.index_select(center_orbital,0,index_neigh),cut_distances)
        center_orbital=torch.index_add(zero_orbital,0,index_center,weight_orbital)
        contracted_orbital=torch.einsum("ijk,jkm->ijm",center_orbital,contracted_coeff)
        density=torch.einsum("ijm,ijm ->im",contracted_orbital,contracted_orbital)+center_coeff
        return density,center_orbital

    def log_tensor_memory(tensor, name):
        print(f"{name}: size={tensor.size()}, allocated memory={torch.cuda.memory_allocated()} bytes")
