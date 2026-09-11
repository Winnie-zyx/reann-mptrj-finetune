import torch
import numpy as np
import torch.distributed as dist

class DataLoader():
    def __init__(self,image,label,numatoms,index_ele,atom_index,shifts,volume,batchsize,min_data_len=None,shuffle=True):
        self.image=image
        self.label=label
        self.index_ele=index_ele
        self.numatoms=numatoms
        self.atom_index=atom_index
        self.shifts=shifts
        self.volume=volume
        self.batchsize=batchsize
        self.end=self.image.shape[0]
        self.shuffle=shuffle               # to control shuffle the data
        if self.shuffle:
            self.shuffle_list=torch.randperm(self.end)
        else:
            self.shuffle_list=torch.arange(self.end)
        if not min_data_len:
            self.min_data=self.end
        else:
            self.min_data=min_data_len
        self.length=int(np.ceil(self.min_data/self.batchsize))
        #print(dist.get_rank(),self.length,self.end)
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        if self.ipoint < self.min_data:
            index_batch=self.shuffle_list[self.ipoint:min(self.end,self.ipoint+self.batchsize)]
            key_list = []
            key_list0 = []
            key_list0 = index_batch.tolist()
            key_list = [x + 1 for x in key_list0]
            species=self.index_ele.index_select(0,index_batch)
            last_valid_index = (species != -1).cumsum(dim=1).argmax(dim=1)
            max_value = torch.max(last_valid_index)
            speciessss = species[:, :(max_value.item() + 1)]
            coordinates=self.image.index_select(0,index_batch)
            coordinatessss=coordinates[:,:speciessss.size(-1),:]
            abprop=(label.index_select(0,index_batch) for label in self.label)
            lengths_shifts = max(self.shifts[key].size(0) for key in key_list)
            lengths_atom = max(self.atom_index[key].size(1) for key in key_list)
            lengths_batch = index_batch.size(0)
            shiftssss=torch.empty((lengths_batch,int(lengths_shifts),3))
            shiftssss.fill_(float('-inf'))
            atom_indexxxx=torch.zeros((2,lengths_batch,int(lengths_atom)),dtype=torch.long)
            for i, key in enumerate(key_list):
                shiftssss[i,:self.shifts[key].size(0), :] = self.shifts[key]
                atom_indexxxx[:,i,:self.atom_index[key].size(1)] = self.atom_index[key]
            volume=self.volume.index_select(0,index_batch)
            numatoms=self.numatoms.index_select(0,index_batch)
            self.ipoint+=self.batchsize
            del coordinates,species
            return abprop,coordinatessss,numatoms,speciessss,atom_indexxxx,shiftssss,volume
        else:
            # if shuffle==True: shuffle the data 
            if self.shuffle:
                self.shuffle_list=torch.randperm(self.end)
            #print(dist.get_rank(),"hello")
            raise StopIteration
