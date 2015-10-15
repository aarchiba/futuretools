#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging import debug

debug = print

import mpi4py.rc
print("threaded:",mpi4py.rc.threaded)
print("thread_level:",mpi4py.rc.thread_level)

import mpi_executor

def square(x):
    return x**2

def cube(x):
    return x**3

if __name__=='__main__':
    M = mpi_executor.MPIExecutor()
    logging.basicConfig(filename='process-{0}.log'.format(M.rank),
                        level=logging.DEBUG)
    debug("Started initial setup")
    if M.is_master():
        debug("Sending jobs to master")
        M.submit(square, -1)
        M.submit(cube, -1)
        print(list(M.map(square, range(100))))
        print(list(M.map(cube, range(100))))
    else:
        M.wait()
