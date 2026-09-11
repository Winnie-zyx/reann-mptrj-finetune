import torch
import numpy as np
import ase.io.vasp
from ase import Atoms, units
from mace.calculators import MACECalculator
import time
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms
from ase.io import read, write
import os
def calculate_single_energy(filename):
    """计算单个POSCAR文件的能量"""
    try:
        # 读取POSCAR文件
        atoms = read(filename, format='vasp')

        # 设置约束 - 固定所有原子
        indices_to_fix = list(range(0, len(atoms)))
        constraint = FixAtoms(indices=indices_to_fix)
        atoms.set_constraint(constraint)

        # 设置计算器
        device = 'cpu'
        period = [1, 1, 1]
        #calc = REANN(properties=['energy', 'stress', 'force'],
        #            nn="REANN_PES_DOUBLE.pt",
        #            device=device,
        #            period=period)
        calc = MACECalculator(
            model_paths="2023-12-03-mace-128-L1_epoch-199.model",  # 改成你真实的模型名/路径
            device="cpu",
            default_dtype="float32")

        atoms.calc = calc

        # 计算并返回能量
        energy = round(atoms.get_potential_energy(), 3)
        return energy, None
    except Exception as e:
        return None, str(e)

def batch_calculate_energies():
    """批量计算POSCAR_00到POSCAR_06的能量"""
    # 存储所有结果
    results = {}

    # 遍历00到06的文件
    for i in range(7):  # 0到6共7个文件
        filename = f'../conf/POSCAR_{i:02d}'  # 格式化文件名，确保是两位数

        # 检查文件是否存在
        if not os.path.exists(filename):
            print(f"警告: 文件 {filename} 不存在，已跳过")
            results[filename] = ("文件不存在", None)
            continue

        # 计算能量
        print(f"正在处理 {filename}...")
        start_time = time.time()
        energy, error = calculate_single_energy(filename)
        end_time = time.time()

        # 处理结果
        if error is None:
            print(f"{filename} 的能量计算完成，耗时: {end_time - start_time:.2f}秒")
            results[filename] = (energy, None)
        else:
            print(f"{filename} 计算出错: {error}")
            results[filename] = (None, error)

    return results

if __name__ == "__main__":
    print("开始批量计算POSCAR文件能量...")
    start_total = time.time()

    # 执行批量计算
    results = batch_calculate_energies()
 
    # 输出汇总结果
    print("\n===== 能量计算结果汇总 =====")
    for filename, (energy, error) in results.items():
        if error is None:
            print(f"{filename}: {energy} eV")
        else:
            print(f"{filename}: 计算失败 - {error}")

    end_total = time.time()
    print(f"\n所有计算完成，总耗时: {end_total - start_total:.2f}秒")

