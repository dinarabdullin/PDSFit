"""Manage multiprocessing options."""


def set_mpi(mpi_support):
    global run_with_mpi
    if mpi_support == 0:
        run_with_mpi = False
        import multiprocessing
        multiprocessing.freeze_support()
    elif mpi_support == 1:
        run_with_mpi = True
        import mpi4py
        from mpi4py import MPI


def get_mpi():
     return run_with_mpi