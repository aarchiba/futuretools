#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MPIExecutor.

An MPIExecutor maintains two threads internally, and is handed work
from the main program. Submitted jobs go on _work_queue; the dispatcher
thread watches this queue and _idle_nodes. When both a job and an idle
node are available, the job is submitted to the idle node, and its
details are put on the _new_in_progress queue. The dispatcher thread
sends an MPI message to the master process (itself). The collector thread
is normally waiting for MPI messages from either itself or any of the
_in_progress jobs. If it gets a message from itself, it empties the
_new_in_progress queue into the _in_progress list. If it gets a message
from one of the _in_progress jobs, then that job is finished or raised
an exception; set the state of the appropriate future and add the machine
to the _idle_nodes queue.

Jobs are assigned a unique integer id at the moment of submission, and
_jobs is a dictionary mapping ids to work items (future, function, arguments).
Access is protected by _jobs_lock.
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

debug = print

class _function_wrapper(object):
    def __init__(self, function):
        self.function = function


def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what "
                       "function to apply.")


class WorkItem(object):
    def __init__(self, jobid, future, fn, args, kwargs):
        self.jobid = jobid
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
        debug("Setting up MPIExecutor with rank {0} and size {1}"
              .format(self.rank, self.size))
        self.functions = [_error_function] * self.size
        if self.size == 0:
            raise ValueError("Tried to create an MPI executor, but there "
                             "was only one MPI process available. "
                             "Need at least two.")
        if self.is_master():
            debug("Setting up master.")
            self._work_queue = queue.Queue()
            self._new_in_progress = queue.Queue()
            self._in_progress = [None for i in range(self.size)]
            self._idle_nodes = queue.Queue()
            for i in range(self.size):
                self._idle_nodes.put(i)
            self._job_counter = 0
            self._jobs = {}
            self._jobs_lock = threading.Lock()
            self._dispatcher = threading.Thread(target=self._dispatcher_loop)
            self._dispatcher.start()
            self._collector = threading.Thread(target=self._collector_loop)
            self._collector.start()

    def is_master(self):
        """
        Is the current process the master?
        """
        return self.rank == 0

    def _dispatcher_loop(self):
        debug("Dispatcher loop started.")
        if not self.is_master():
            raise RuntimeError("Worker node told to dispatch jobs.")
        while True:
            debug("Waiting for work.")
            job = self._work_queue.get(block=True)
            debug("Got task {0}: {1}(*{2},**{3})."
                  .format(job.jobid, job.fn, job.args, job.kwargs))
            node = self._idle_nodes.get(block=True)
            debug("Found node {0} free.".format(node))
            if job.f.set_running_or_notify_cancel():
                task = (job.jobid, job.args, job.kwargs)
                if job.fn is not self.functions[node]:
                    debug("Master replacing worker {0} function with {1}."
                          .format(node,function))
                    self.comm.send(F, node+1)
                    self.functions[node] = job.fn
                debug("Dispatcher calling MPI")
                self.comm.isend(task, dest=node+1, tag=node)
            else:
                # job was cancelled before we could start it
                self._idle_nodes.put(node)
    def _collector_loop(self):
        debug("Collector loop started.")
        if not self.is_master():
            raise RuntimeError("Worker node told to collect jobs.")
        while True:
            debug("Waiting for results.")
            debug("Collector calling MPI")
            status = MPI.Status()
            # Receive input from workers.
            result = self.comm.recv(source=MPI.ANY_SOURCE,
                                    tag=MPI.ANY_TAG, status=status)
            worker = status.source
            debug("Got result from worker {0}, marking it idle"
                  .format(worker))
            node = status.tag
            self._idle_nodes.put(node)
            # FIXME: handle exceptions
            jobid, value = result
            with self._jobs_lock:
                task = self._jobs.pop(jobid)
            task.f.set_result(value)

    def submit(self, fn, *args, **kwargs):
        i = self._job_counter
        self._job_counter += 1
        debug("Submitting job {0}: {1}(*{2},**{3})."
              .format(i, fn, args, kwargs))
        f = concurrent.futures.Future()
        w = WorkItem(i,f,fn,args,kwargs)
        with self._jobs_lock:
            self._jobs[i] = w
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
            jobid, args, kwargs = task
            result = self.function(*args,**kwargs)
            debug("Worker {0} sending answer {1} with tag {2}."
                  .format(self.rank, result, status.tag))
            self.comm.isend((jobid,result), dest=0, tag=status.tag)
