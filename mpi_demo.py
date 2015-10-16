#!/usr/bin/env python
# -*- coding: utf-8 -*-

# You should probably run this as "mpirun -n 8 python mpi_demo.py"

import traceback
import logging
from logging import debug

import mpi_executor

def square(x):
    return x**2

def cube(x):
    return x**3

def bonk(x):
    if x==7:
        raise ValueError("bonk")
    else:
        return x
if __name__=='__main__':
    M = mpi_executor.MPIExecutor()
    logging.basicConfig(filename='process-{0}.log'.format(M.rank),
                        level=logging.DEBUG)
    debug("Started initial setup")
    if M.is_master():
        debug("Sending jobs to master")
        M.submit(square, -1)
        M.submit(cube, -1)
        print(list(M.map(square, range(10))))
        print(list(M.map(cube, range(10))))
        try:
            print(list(M.map(bonk, range(10))))
        except ValueError as e:
            print("Caught a ValueError:")
            traceback.print_exc()
        M.shutdown()
    else:
        M.wait()
