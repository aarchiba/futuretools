#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MPIExecutor.

An MPIExecutor maintains one thread internally, which must make all
MPI calls. This thread is handed work from the main program.
When both a job and an idle
node are available, the job is submitted to the idle node. Then the
thread checks for incoming messages (completed results); any such
are handed back to the Future and the node is marked as clear. The
process then waits a short time (1 to 50 ms, falling off exponentially
if nothing happens) before repeating the queries.

Polling is necessary because (at least) OpenMPI does not support
being called from multiple threads, and receiving messages blocks
the whole thread; there is no way for it to react to newly-submitted
jobs unless they generate MPI messages, and any such messages would
have to come from a different process. MPI also does not support a
timeout on recv.

Jobs are assigned a unique integer id at the moment of submission, and
_jobs is a dictionary mapping ids to work items (future, function, arguments).
Access is protected by _jobs_lock.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["MPIExecutor"]

import time

# If mpi4py is installed, import it.
import mpi4py.rc
mpi4py.rc.threaded = True
mpi4py.rc.thread_level = 'funneled'
from mpi4py import MPI

import traceback
from logging import debug
#debug = print
#debug = lambda x: None
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

class _exception_wrapper(object):
    def __init__(self, exc):
        self.exc = exc

class WorkItem(object):
    def __init__(self, jobid, future, fn, args, kwargs):
        self.jobid = jobid
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

# Hack to embed stringification of remote traceback in local traceback
# from python 3.5's ProcessPoolExecutor

class _RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb
    def __str__(self):
        return self.tb

class _ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb
    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)

def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc

class MPIExecutor(concurrent.futures.Executor):
    def __init__(self, comm=None):
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
            self._in_progress = [None for i in range(self.size)]
            self._job_counter = 0
            self._jobs = {}
            self._jobs_lock = threading.Lock()
            self.min_interval = 1e-3
            self.max_interval = 5e-2
            self._stop_now = threading.Event()
            self._dispatcher = threading.Thread(target=self._dispatcher_loop)
            self._dispatcher.start()

    def is_master(self):
        """
        Is the current process the master?
        """
        return self.rank == 0

    def _dispatcher_loop(self):
        debug("Dispatcher loop started.")
        if not self.is_master():
            raise RuntimeError("Worker node told to dispatch jobs.")
        job = None
        interval = self.min_interval
        while True:
            debug("Top of dispatcher.")
            did_something = False

            if job is None:
                with self._jobs_lock:
                    # Are we waiting for jobs to finish? If not, might as well
                    # block until new ones submitted.
                    todo = len(self._jobs)>0
                if not todo and self._stop_now.is_set():
                    break
                try:
                    if todo:
                        job = self._work_queue.get(timeout=interval)
                    else:
                        job = self._work_queue.get()
                except queue.Empty:
                    job = None
                else:
                    debug("Got task {0}: {1}(*{2},**{3})."
                          .format(job.jobid, job.fn, job.args, job.kwargs))
                    if job.fn is None:
                        # just a wakeup call
                        with self._jobs_lock:
                            self._jobs.pop(job.jobid)
                        job = None

            node = None
            for i, j in enumerate(self._in_progress):
                if j is None:
                    node = i
                    debug("Found node {0} free.".format(node))
                    break
            if node is None:
                if job is None:
                    # If no job to do then we already slept in .get()
                    pass
                else:
                    # give it time for a node to free up
                    debug("Sleeping for {0} sec.".format(interval))
                    time.sleep(interval)

            if (job is not None
                and node is not None):
                debug("about to poke future {0}".format(job.future))
                if job.future.set_running_or_notify_cancel():
                    debug("Trying submission.")
                    task = (job.jobid, job.args, job.kwargs)
                    if job.fn is not self.functions[node]:
                        debug("Master replacing worker {0} function with {1}."
                              .format(node,job.fn))
                        self.comm.send(_function_wrapper(job.fn), node+1)
                        self.functions[node] = job.fn
                        debug("new function sent")
                    debug("Dispatcher calling MPI")
                    self.comm.isend(task, dest=node+1, tag=node)
                    self._in_progress[node] = job
                    job = None
                else:
                    job = None
                    debug("job appears to have been cancelled")
                did_something = True

            debug("Checking MPI status")
            status = MPI.Status()
            # Receive input from workers.
            debug("Probing MPI for messages")
            if self.comm.Iprobe(source=MPI.ANY_SOURCE,
                                tag=MPI.ANY_TAG):
                debug("Receiving message.")
                jobid, result = self.comm.recv(source=MPI.ANY_SOURCE,
                                               tag=MPI.ANY_TAG, status=status)
                worker = status.source-1
                debug("Got result from worker {0}, marking it idle"
                      .format(worker))
                self._in_progress[worker] = None
                with self._jobs_lock:
                    done_job = self._jobs.pop(jobid)
                if isinstance(result, _exception_wrapper):
                    done_job.future.set_exception(result.exc)
                else:
                    done_job.future.set_result(result)
                did_something = True
            else:
                debug("No messages.")
            if did_something:
                interval = self.min_interval
            else:
                interval = min(interval*1.1, self.max_interval)
        debug("Dispatcher loop exited.")
        for node in range(self.size):
            self.comm.send(_close_pool_message(), node+1)


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
            debug("Worker {0} waiting for task.".format(self.rank-1))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            debug("Worker {0} got task {1} with tag {2}."
                  .format(self.rank-1, task, status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                debug("Worker {0} told to quit.".format(self.rank-1))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, _function_wrapper):
                self.function = task.function
                debug("Worker {0} replaced its task function: {1}."
                      .format(self.rank-1, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            jobid, args, kwargs = task
            try:
                result = self.function(*args,**kwargs)
            except Exception as exc:
                exc = _ExceptionWithTraceback(exc, exc.__traceback__)
                result = _exception_wrapper(exc)
            debug("Worker {0} sending answer {1} with tag {2}."
                  .format(self.rank-1, result, status.tag))
            self.comm.isend((jobid,result), dest=0, tag=status.tag)

    def shutdown(self, wait=True):
        debug("shutdown called")
        self._stop_now.set()
        self.submit(None)
        if wait:
            debug("waiting for dispatcher to finish")
            self._dispatcher.join()
