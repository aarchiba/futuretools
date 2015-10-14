#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import mpi_executor

def square(x):
    return x**2

def cube(x):
    return x**3

if __name__=='__main__':
    M = mpi_executor.MPIExecutor()
    logging.basicConfig(filename='process-{0}.log'.format(M.rank),
                        level=logging.DEBUG)
    if M.is_master():
        print(list(M.map(square, range(100))))
        print(list(M.map(cube, range(100))))
    else:
        M.wait()
