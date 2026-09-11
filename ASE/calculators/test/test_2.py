# This is an example script to show how to obtain the energy and force by invoking the potential saved by the training .
# Typically, you can read the structure,mass, lattice parameters(cell) and give the correct periodic boundary condition (pbc) and t    he index of each atom. All the information are required to store in the tensor of torch. Then, you just pass these information to t    he calss "pes" that will output the energy and force.

import numpy as np
import torch
#from gpu_sel import *
# used for select a unoccupied GPU
#gpu_sel()
# gpu/cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# same as the atomtype in the file input_density
atomtype=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
#load the serilizable model
pes=torch.jit.load("REANN_PES_DOUBLE.pt")
# FLOAT: torch.float32; DOUBLE:torch.double for using float/double in inference
pes.to(device).to(torch.double)
# set the eval mode
pes.eval()
pes=torch.jit.optimize_for_inference(pes)
# save the lattic parameters
cell=np.zeros((3,3),dtype=np.float64)
strss=np.zeros((3,3),dtype=np.float64)
period_table=torch.tensor([1,1,1],dtype=torch.double,device=device)   # same as the pbc in the periodic boundary condition

rmse=torch.zeros(3,dtype=torch.double,device=device)
mae=torch.zeros(3,dtype=torch.double,device=device)
#i=0
#sig=0

fe=open("test-ene.dat",'w+')
fe.write('# point DFT ML\n')
ff=open("test-force_ML.dat",'w+')
bb=open("test-force_DFT.dat",'w+')
ss=open("test-stress_ML.dat",'w+')
tt=open("test-stress_DFT.dat",'w+')

#ff.write('# point num abene force_rmse\n')
#with open("./configuration",'r') as f1:
#natom=[]
#with open("/public/home/zyx/dataset/S.P.Ong/REANN/cal1221/t=0/LYC/1/number/natom_test", 'r', encoding='utf-8') as na:
#    natom=[_.rstrip('\n') for _ in na.readlines()]
#print(natom)
#with open('/home/home/zyx/LYC/rmse_test/3_1/pred_data.txt','a+') as fp:
#    f.writelines(["Configuration",'\t',"Energy",'\t',"Force",'\n'])
#    f.close
npoint=0
nforce=0
#with open("/public/home/zyx/dataset/S.P.Ong/REANN/cal1221/t=0/LYC/1/data/test/configuration",'r') as f1:
with open("configuration_10",'r') as f1:
    #with open("/public/home/zyx/dataset/S.P.Ong/REANN/cal1221/t=0/LYC/1/data/test/config_error",'r') as f1:
        while True:
            string=f1.readline()
            id_line=string
            if not string: break
            string=f1.readline()
            cell[0]=np.array(list(map(float,string.split())))
            string=f1.readline()
            cell[1]=np.array(list(map(float,string.split())))
            string=f1.readline()
            cell[2]=np.array(list(map(float,string.split())))
            string=f1.readline()
#            strss1=string.split()
#            tmp1=list(map(float,tmp[2:8]))
#            strss[0]=np.array(list(map(float,string.split()[2:5])))
#            strss[1]=np.array(list(map(float,string.split()[5:8])))
#            strss[2]=np.array(list(map(float,string.split()[8:11])))
            species=[]
            cart=[]
            abforce=[]
            abstress=[]
            mass=[]
            while True:
                string=f1.readline()
                if "abprop" in string: break
                tmp=string.split()
                tmp1=list(map(float,tmp[2:8]))
                cart.append(tmp1[0:3])
                abforce.append(tmp1[3:6])
##                abstress.append(tmp1[:])
                mass.append(float(tmp[1]))
                species.append(atomtype.index(tmp[0]))
            abene=float(string.split()[1])
            strss[0]=np.array(list(map(float,string.split()[2:5])))
            strss[1]=np.array(list(map(float,string.split()[5:8])))
            strss[2]=np.array(list(map(float,string.split()[8:11])))
            abstress.append(strss[0:3])
            abene=torch.from_numpy(np.array([abene])).to(device)
            species=torch.from_numpy(np.array(species)).to(device)  # from numpy array to torch tensor
            cart=torch.from_numpy(np.array(cart)).to(device).to(torch.double)  # also float32/double
            mass=torch.from_numpy(np.array(mass)).to(device).to(torch.double)  # also float32/double
            abforce=torch.from_numpy(np.array(abforce)).to(device).to(torch.double)  # also float32/double
            abstress=torch.from_numpy(np.array(abstress)).to(device).to(torch.double)
#            print(abstress)
            tcell=torch.from_numpy(cell).to(device).to(torch.double)  # also float32/double
            #print(period_table,cart,tcell,species,mass)
#            npoint+=1
            try:
                energy,stress,force=pes(period_table,cart,tcell,species,mass)
                energy=energy.detach()
                force=force.detach()
                stress=stress.detach()
                #print(energy,'\n',stress,'\n'+'-'*40)
                #stress=stress.detach()
                #ee=float(energy.detach())
                #aa=float(abene)
                #sig=sig+(ee-aa)**2
                fff=force.detach().numpy()
                bbb=abforce.numpy()
                sss=stress.detach().numpy()
                ttt=abstress.numpy()
                print(sss)
                print(energy)
                print(fff)

                #rmse[0]+=torch.sum(torch.square(energy-abene))
                #mae[0]+=torch.sum(torch.abs(energy-abene)/natom[])
                fe.write("%4d%18.8f%18.8f\n" %(npoint+1,abene,energy))
                rmse[1]+=torch.sum(torch.square(force-abforce))
                rmse[2]+=torch.sum(torch.square(stress-abstress))
                
                ncf=0
                for i in fff.flatten('A'):
                  nforce+=1
                  ncf+=1
                  ff.write("%18.8f\n" %(i))
                for i in bbb.flatten('A'):
                  bb.write("%18.8f\n" %(i))
                for i in sss.flatten('A'):
                  ss.write("%18.8f\n" %(i))
                for i in ttt.flatten('A'):
                  tt.write("%18.8f\n" %(i))
                npoint+=1 

                rmse[0]+=torch.sum(torch.square((energy-abene)/ncf*3))
                mae[0]+=torch.sum(torch.abs(energy-abene)/ncf*3)
                mae[1]+=torch.sum(torch.abs(force-abforce))
                mae[2]+=torch.sum(torch.abs(stress-abstress))
                #natom=ncf
                #print(ncf/3)




            except:
                print('wrong config {0} {1}'.format(npoint,id_line))   
                

            #npoint+=1
#            rmse=rmse.detach().cpu().numpy()
#            print(np.sqrt(rmse[0]/npoint))
#            print(np.sqrt(rmse[1]/npoint/192/3))
#            print(npoint)


            
#            chafb=ff-bb
#            chafb2=np.power(chafb,2)
#            print(chafb2)
#            chafb2_sum=np.sum(chafb2,axis=0))
#            chafb_1=pow(chafb2_sum(0),0.5)
#            chafb_2=pow(chafb2_sum(1),0.5)
#            chafb_3=pow(chafb2_sum(2),0.5)
#            ff1=ff1+chafb_1**2
#            ff2=ff2+chafb_2**2
#            ff3=ff3+chafb_3**2
#            sig=sig+(ee-aa)**2
#            i=i+1



         #print(float(abene))
         #print(ee)
#         with open('/public/home/zyx/LYC/REANN/test1201/1800/cal1010_1/result.dat','a+') as f:
#            fp.write(str(ee),str(ff)+'\n')
#            with open('/home/home/zyx/LYC/rmse_test/test_522/test_data.txt','a+') as ft:
#                ft.write(str(aa),str(bb)+'\n')


#rmse_ee=(sig/i)**0.5
#rmse_ff1=(sib/i)**0.5
#r=float(rmse)
#with open('/home/home/zyx/LYC/rmse_test/test_522/rmse.dat','a+') as f:
#    f.write(str(rmse_ee),str(rmse_ff))

rmse=rmse.detach().cpu().numpy()
#frmse=np.sqrt(rmse[1]/npoint/3)
#print(np.sqrt(rmse[0]/npoint))
#print(np.sqrt(rmse[1]/npoint/480/3))
#print(np.sqrt(rmse[1]/nforce))
#print(np.sqrt(rmse[2]/npoint/9))
#print(frmse)

a1=np.sqrt(rmse[0]/npoint)
a2=np.sqrt(rmse[1]/nforce)
a3=np.sqrt(rmse[2]/npoint/9)

b1=np.abs(mae[0]/npoint)
b2=np.abs(mae[1]/nforce)
b3=np.abs(mae[2]/npoint/9)

with open('rmse.dat','a+') as rmse:
    rmse.write("rmse\n")
    rmse.write("energy  "+"%18.8f\n" %(a1))
    rmse.write("force   "+"%18.8f\n" %(a2))
    rmse.write("stress  "+"%18.8f\n" %(a3))
    rmse.write("mae\n")
    rmse.write("energy  "+"%18.8f\n" %(b1))
    rmse.write("force   "+"%18.8f\n" %(b2))
    rmse.write("stress  "+"%18.8f\n" %(b3))


print(npoint)



