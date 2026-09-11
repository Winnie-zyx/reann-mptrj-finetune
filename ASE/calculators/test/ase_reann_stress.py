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

#import sys
#sys.path.append("/public/home/zyx/LYC/REANN/REANN-stress-realease/reann/ASE/calculators/")
#from reann_stress import REANN  #the path of reann_stress.py
from ase.calculators.reann_stress import REANN
# 定义构型文件路径
config_file = "datpath.extxyz"  # 替换为你的构型文件路径

with open(config_file, 'r') as fileobj:
    configuration = extxyz.read_extxyz(fileobj, index=slice(0, 10))

    # 定义 计算器
#--------------the type of atom--------------
    atomtype=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'I', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    #atomtype = ['Cu','Ce','O','C']
#-----------------the device is cpu or gpu( cpu is default)---------------
    device='cpu'
#-------------------------pbc([1,1,1] is default)---------------------
    period=[1,1,1]
#---------------nn file('EANN_PES_DOUBLE.pt' is default in eann,'REANN_PES_DOUBLE.pt' is default in reann)----------------------------
    nn = 'REANN_PES_DOUBLE.pt'
#----------------------eann (if you use EANN package )--------------------------------
#atoms.calc = EANN(device=device,atomtype=atomtype,period=period,nn = nn)
#----------------------------reann (if you use REANN package ****recommend****)---------------------------------
#atoms.calc = REANN(device=device,atomtype=atomtype,period=[1,1,0],nn = 'REANN_PES_DOUBLE.pt')



    #def calculate(self,atoms=None, properties=['energy'],
    #              system_changes=all_changes):

    calc = REANN(atomtype=atomtype, device=device, properties=['energy','stress','force'], period=period, nn = "REANN_PES_DOUBLE.pt")

# 遍历每个构型
    for atoms in configuration:
    # 关联计算器
        atoms.set_calculator(calc)
    
        energy = atoms.get_potential_energy(apply_constraint=False)
        force = atoms.get_forces()
    # 计算应力
        stress = atoms.get_stress(voigt=False)
        #stress = atoms.get_stress(voigt=False) / units.GPa  # 转换为 GPa 单位
        print("Stress:", stress)
        print("Energy:", energy)
        print("Force:", force)
