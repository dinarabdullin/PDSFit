def set_mpi(mpi_support):
    global run_with_mpi
    if mpi_support == 0:
        run_with_mpi = False
    elif mpi_support == 1:
        run_with_mpi = True


def get_mpi():
     return run_with_mpi