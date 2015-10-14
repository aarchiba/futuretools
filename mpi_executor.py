#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MPIExecutor.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MPIExecutor"]

# If mpi4py is installed, import it.
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from logging import debug
import threading
import concurrent.futures
try:
    import queue
except ImportError: # py < 3
    import Queue as queue

class _close_pool_message(object):
    def __repr__(self):
        return "<Close pool message>"


class _function_wrapper(object):
    def __init__(self, function):
        self.function = function


def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what "
                       "function to apply.")


class WorkItem(object):
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class MPIExecutor(concurrent.futures.Executor):
    def __init__(self, comm=None):
        if MPI is None:
            raise ImportError("Please install mpi4py")
        self.comm = MPI.COMM_WORLD if comm is None else comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() - 1
        self.functions = [_error_function] * self.size
        if self.size == 0:
            raise ValueError("Tried to create an MPI executor, but there "
                             "was only one MPI process available. "
                             "Need at least two.")
        if self.is_master():
            debug("Setting up master.")
            self._work_queue = queue.Queue()
            self._in_progress = [None for i in range(self.size)]
            self._dispatcher = threading.Thread(target=self._master_loop)
            self._dispatcher.start()

    def is_master(self):
        """
        Is the current process the master?
        """
        return self.rank == 0

    def _master_loop(self):
        debug("Master loop started.")
        if not self.is_master():
            raise RuntimeError("Worker node told to dispatch jobs.")
        while True:
            debug("Waiting for work.")
            job = self._work_queue.get(block=True)
            debug("Got task {0}.".format(job))
            idle = None
            done_job = None
            for (i,v) in enumerate(self._in_progress):
                if v is None:
                    idle = i
            if idle is None:
                status = MPI.Status()
                # Receive input from workers.
                result = self.comm.recv(source=MPI.ANY_SOURCE,
                                        tag=MPI.ANY_TAG, status=status)
                worker = status.source
                i = status.tag
                done_job = self._in_progress[i]
                self._in_progress[i] = None
                done_job.future.set_result(result)
            if job.set_running_or_notify_cancel():
                task = (job.args, job.kwargs)
                if job.fn is not self.functions[i]:
                    debug("Master replacing worker {0} function with {1}."
                          .format(i,function))
                    self.comm.send(F, i+1)
                    self.functions[i] = job.fn
                self.comm.isend(task, dest=i+1, tag=i)
                self._in_progress[i] = job


    def submit(self, fn, *args, **kwargs):
        f = concurrent.futures.Future()
        w = WorkItem(f,fn,args,kwargs)
        self._work_queue.put(w)
        return f
    submit.__doc__ = concurrent.futures.Executor.submit.__doc__

    def wait(self):
        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        status = MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            debug("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            debug("Worker {0} got task {1} with tag {2}."
                  .format(self.rank, task, status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                debug("Worker {0} told to quit.".format(self.rank))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, _function_wrapper):
                self.function = task.function
                debug("Worker {0} replaced its task function: {1}."
                      .format(self.rank, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            args, kwargs = task
            result = self.function(*args,**kwargs)
            debug("Worker {0} sending answer {1} with tag {2}."
                  .format(self.rank, result, status.tag))
            self.comm.isend(result, dest=0, tag=status.tag)
