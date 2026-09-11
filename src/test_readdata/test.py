from read_data import *





start_table=0
#floder_train=floder+"train/"
floder_test="/public/home/zyx/reann/REANN-stress/REANN-stress-realease/reann/src/test_readdata/"
# obtain the number of system
floderlist=[floder_test]

if start_table==0 or start_table==1:
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pot,force,atomic_number=  \
    Read_data(floderlist,1,start_table=start_table)
    print(atom)
    print(atomic_number)
elif start_table==5: # pot contains static stress tensor
    numpoint,atom,mass,numatoms,scalmatrix,period_table,coor,pot_stress,force=  \
    Read_data(floderlist,10,start_table=start_table) # 10 = energy(1) + 9(stress)
    pot_stress=torch.from_numpy(np.array(pot_stress,dtype=np_dtype))
    pot=pot_stress[:,0].numpy()
    stress=pot_stress[:,1:]#/1602.1766208#.reshape(-1,3,3)
else:
    print("error")

#import
