#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from logging import debug
import threading
import concurrent.futures
try:
    import queue
except ImportError: # py < 3
    import Queue as queue

def _do_nothing():
    pass

class IpyExecutor(concurrent.futures.Executor):
    def __init__(self, client):
        concurrent.futures.Executor.__init__(self)
        self.client = client
        self.view = self.client.load_balanced_view()
        self.to_run = queue.Queue()
        self._in_progress = []
        self._in_progress_lock = threading.Lock()
        self.min_interval = 1e-3 # s
        self.max_interval = 5e-2 # s
        # this is used to wait for changes in the number of busy nodes
        # either a node goes free or a node is newly in use; it is also
        # notified when shutting down. Both threads wait on this at
        # different times.
        self._recheck_nodes = threading.Condition()
        # shutdown sets this; no more jobs can be submitted and the
        # dispatcher exits when its queue is empty.
        self._time_to_stop = threading.Event()
        # the dispatcher sets this when there are no more jobs in the queue
        # (and none can be submitted because _time_to_stop is set); the
        # collector exits when no more jobs are in progress
        self._no_more_jobs = threading.Event()

        self._dispatcher_thread = threading.Thread(target=self._dispatcher)
        self._dispatcher_thread.start()
        self._collector_thread = threading.Thread(target=self._collector)
        self._collector_thread.start()

    def submit(self, fn, *args, **kwargs):
        if self._time_to_stop.is_set() and fn is not None:
            raise ValueError("Job submitted after shutdown called")
        f = concurrent.futures.Future()
        self.to_run.put((f,fn,args,kwargs))
        return f

    def _node_free(self):
        with self._in_progress_lock:
            return len(self._in_progress)<len(self.view)

    def _dispatcher(self):
        while True:
            if (self._time_to_stop.is_set()
                and self.to_run.empty()):
                self._no_more_jobs.set()
                # Wake up collector
                with self._recheck_nodes:
                    self._recheck_nodes.notify_all()
                break
            f, fn, args, kwargs = self.to_run.get()
            if fn is None:
                continue
            with self._recheck_nodes:
                self._recheck_nodes.wait_for(self._node_free)
            if f.set_running_or_notify_cancel():
                r = self.view.apply_async(fn, *args, **kwargs)
                r.future = f
                with self._in_progress_lock:
                    self._in_progress.append(r)
                with self._recheck_nodes:
                    self._recheck_nodes.notify_all()

    def _collector(self):
        # It seems that zmq requires polling, so we're stuck
        # polling for finished tasks.
        interval = self.min_interval
        while True:
            # Block until either there's a job or we're told to stop
            def a_job():
                if self._no_more_jobs.is_set():
                    return True
                with self._in_progress_lock:
                    return len(self._in_progress)>0
            with self._recheck_nodes:
                self._recheck_nodes.wait_for(a_job)
            if self._no_more_jobs.is_set():
                with self._in_progress_lock:
                    if not self._in_progress:
                        break

            # Check for any jobs that have finished
            with self._in_progress_lock:
                finished = []
                in_progress = []
                for r in self._in_progress:
                    if r.ready():
                        finished.append(r)
                    else:
                        in_progress.append(r)
                if finished:
                    self._in_progress = in_progress
                del in_progress
            # Pass the results to the Futures
            for r in finished:
                try:
                    value = r.get()
                except BaseException as e:
                    r.future.set_exception(e)
                else:
                    r.future.set_result(value)
            # If some jobs finished, nodes may be free; notify dispatcher
            if finished:
                with self._recheck_nodes:
                    self._recheck_nodes.notify_all()
                interval = self.min_interval
            else:
                # We always start the job with at least one job
                # so if nothing finished we've got one in progress;
                # give it time to finish (backing off exponentially)
                time.sleep(interval)
                interval = min(interval*1.1, self.max_interval)


    def shutdown(self, wait=True):
        self._time_to_stop.set()
        # Wake up dispatcher
        self.submit(None)
        if wait:
            self._no_more_jobs.wait()
            def no_jobs():
                with self._in_progress_lock:
                    return self.to_run.empty() and len(self._in_progress)==0
            with self._recheck_nodes:
                self._recheck_nodes.wait_for(no_jobs)
        # wake the submitter thread if it's waiting for a job
        if wait:
            self._dispatcher_thread.join()
            self._collector_thread.join()
