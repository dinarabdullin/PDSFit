Code was written using:
- python 3.14.5 (including modules time, sys, argparse, math, os, datetime, errno, shutil)
- numpy 2.4.6
- scipy 1.17.1
- matplotlib 3.10.9
- libconf 2.0.1

For MPI application, the following is needed:
- MPI (Windows) / OpenMPI 4.0.5 (Linux)
- mpi4py 4.1.2

For the compilation, the following is needed:
- pyinstaller 6.20.0

1) Run:
pyinstaller --onefile PDSFit.py

2) Edit PDSFit.spec:
exe = EXE(..., [('W ignore', None, 'OPTION')], ...)

3) Run:
pyinstaller --onefile PDSFit.spec