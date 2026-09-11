##
##load REANN.pth, and then change the torch-script in .pt from the initial code to the changed code 

import sys

mod=sys.argv[1]
if mod=='0':
    import pes.script_PES as PES_Normal
if mod=='5':
    import stress.script_PES as PES_Normal
    import lammps.script_PES as PES_Lammps
    print("import stress.script_PES as PES_Normal")
PES_Normal.jit_pes()
PES_Lammps.jit_pes()
