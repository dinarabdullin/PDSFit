Code was written using:
- python 3.8.6 (including modules time, sys, argparse, math, os, datetime, errno, shutil)
- numpy 1.19.3
- scipy 1.5.4
- matplotlib 3.3.3
- libconf 2.0.1

Compilation:
- pyinstaller 4.0
1) Run:
pyinstaller --onefile main.py
2) Edit main.spec:
exe = EXE(..., [('W ignore', None, 'OPTION')], ...)
3) Run:
pyinstaller --onefile main.spec