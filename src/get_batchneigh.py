import os
import torch
import numpy as np
import src.get_neighbour as get_neighbour

def get_batch_neigh(com_coor,scalmatrix,species,period,neigh_atoms,batchsize,cutoff,device):
    ntotpoint=com_coor.shape[0]
    maxnumatom=com_coor.shape[1]
    #shifts=torch.empty((ntotpoint,maxnumatom*neigh_atoms,3),dtype=torch.float16)
    #atom_index=torch.empty((2,ntotpoint,maxnumatom*neigh_atoms),dtype=torch.int16)
    #print('get_batchneigh ntotpoint,maxnumatom,neigh_atoms',ntotpoint,maxnumatom,neigh_atoms,flush=True)
    tmpbatch=1
    maxneigh=0
    shifts = {}
    atom_index = {}
    i=1
    for ipoint in range(1,ntotpoint+1):
        if ipoint<ntotpoint and (scalmatrix[ipoint-1]==scalmatrix[ipoint]).all() and \
        (species[ipoint-1]==species[ipoint]).all() and (period[ipoint-1]==period[ipoint]).all \
        and tmpbatch<batchsize:
            tmpbatch+=1
        else:
            species_=species[ipoint-tmpbatch:ipoint].to(device)
            mask_species_ = (species_ != -1) .to(device)
            speciessss_=species_[mask_species_].view(species_.size(0),-1).to(device)
            cart=com_coor[ipoint-tmpbatch:ipoint].to(device)
            cartttt=cart[:,:((speciessss_.size(-1))+0),:].to(device)
            cell=scalmatrix[ipoint-tmpbatch].to(device)
            pbc=period[ipoint-tmpbatch].to(device)
            tmpindex,tmpshifts,neigh=get_neighbour.neighbor_pairs\
            (pbc, cartttt, speciessss_, cell, cutoff, neigh_atoms)
            atom_index[i] = (tmpindex[:,0,:]).to("cpu")
            shifts[i] = (tmpshifts[0]).to("cpu")
            i=i+1
            k=i
            for j in range(k-1,ipoint) :
                shifts[i]=(tmpshifts[j-k+2]).to("cpu")
                atom_index[i] = (tmpindex[:,j-k+2,:]).to("cpu")
                i=i+1
            torch.cuda.empty_cache()
            tmpbatch=1
    return shifts,atom_index
