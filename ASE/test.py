import torch
import numpy as np
import ase.io.vasp
from ase import Atoms, units
from ase.io import extxyz
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
from ase import Atom

import sys
sys.path.append("/public/home/zyx/LYC/REANN/Equi-NN-main/ASE/calculators/")
from Equi_MPNN import Equi_MPNN  # your_module 是包含了 Equi_MPNN 类的模块
import sys
sys.path.append("/public/home/zyx/LYC/REANN/Equi-NN-main/ASE/")
import getneigh as getneigh
# 定义构型文件路径
config_file = "/public/home/zyx/dataset/S.P.Ong/Equi-NN_LYC1845/stress/output/conf.extxyz"  # 替换为你的构型文件路径
f=open("/public/home/zyx/dataset/S.P.Ong/Equi-NN_LYC1845/stress/output/test-stress_ML.dat",'w+')
# 读取构型文件
with open(config_file, 'r') as fileobj:
    configuration = extxyz.read_extxyz(fileobj, index=slice(0, 184))

# 定义 Equi_MPNN 计算器
    calc = Equi_MPNN(maxneigh=25000, getneigh=getneigh, nn='PES.pt', device='cpu', dtype=torch.float32)

# 遍历每个构型
    for atoms in configuration:
    # 关联计算器
        atoms.set_calculator(calc)
    
    # 计算应力
        stress = atoms.get_stress(voigt=False) / units.GPa  # 转换为 GPa 单位
        #print("Stress:", stress)
        a=stress[0]/(-160.21766)
        b=stress[1]/(-160.21766)
        c=stress[2]/(-160.21766)
        f.write("%18.8f\n" %(a[0]))
        f.write("%18.8f\n" %(a[1]))
        f.write("%18.8f\n" %(a[2]))
        f.write("%18.8f\n" %(b[0]))
        f.write("%18.8f\n" %(b[1]))
        f.write("%18.8f\n" %(b[2]))
        f.write("%18.8f\n" %(c[0]))
        f.write("%18.8f\n" %(c[1]))
        f.write("%18.8f\n" %(c[2]))
        
        #print(a[0])
        #print(b)
        #print(c)
