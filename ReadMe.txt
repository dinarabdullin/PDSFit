Code was written using:
- python 3.8.6 (including modules time, sys, argparse, math, os, datetime, errno, shutil)
- numpy 1.19.3
- scipy 1.5.4
- matplotlib 3.3.3
- libconf 2.0.1

For MPI application, the following is needed:
- MPI (Windows) / OpenMPI 4.0.5 (Linux)
- mpi4py

For the compilation, the following is needed:
- pyinstaller 4.0
1) Run:
pyinstaller --onefile PDSFit.py
2) Edit PDSFit.spec:
exe = EXE(..., [('W ignore', None, 'OPTION')], ...)
3) Run:
pyinstaller --onefile PDSFit.spec