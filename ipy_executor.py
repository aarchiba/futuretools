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
        self._recheck_nodes = threading.Condition()
        self._in_progress = []
        self._in_progress_lock = threading.Lock()
        self.min_interval = 1e-3 # s
        self.max_interval = 5e-2 # s
        self.all_done = threading.Event()
        self.all_done.set()
        self._time_to_stop = threading.Event()
        self._dispatcher_thread = threading.Thread(target=self._dispatcher)
        self._dispatcher_thread.start()
        self._collector_thread = threading.Thread(target=self._collector)
        self._collector_thread.start()

    def submit(self, fn, *args, **kwargs):
        f = concurrent.futures.Future()
        self.all_done.clear()
        self.to_run.put((f,fn,args,kwargs))
        return f

    def _node_free(self):
        with self._in_progress_lock:
            return len(self._in_progress)<len(self.view)

    def _dispatcher(self):
        while True:
            if self._time_to_stop.is_set():
                break
            f, fn, args, kwargs = self.to_run.get()
            if fn is None:
                continue
            with self._recheck_nodes:
                while not self._node_free():
                    self._recheck_nodes.wait()
            if f.set_running_or_notify_cancel():
                r = self.view.apply_async(fn, *args, **kwargs)
                r.future = f
                with self._in_progress_lock:
                    self._in_progress.append(r)

    def _collector(self):
        # It seems that zmq requires polling, so we're stuck
        # polling for finished tasks.
        interval = self.min_interval
        while True:
            with self._in_progress_lock:
                finished = []
                in_progress = []
                for r in self._in_progress:
                    if r.ready():
                        finished.append(r)
                    else:
                        in_progress.append(r)
                self._in_progress = in_progress
            for r in finished:
                try:
                    r.future.set_result(r.get())
                except Exception as e:
                    r.future.set_exception(e)
            if finished:
                qs = self.view.queue_status()
                idle = [n for n in qs if n!='unassigned' and qs[n]['tasks']==0]
                with self._recheck_nodes:
                    self._recheck_nodes.notify_all()
                interval = self.min_interval
            else:
                interval = min(interval*1.1, self.max_interval)
            with self._in_progress_lock:
                if not self._in_progress:
                    self.all_done.set()
            if self._time_to_stop.is_set():
                break
            time.sleep(interval)

    def shutdown(self, wait=True):
        if wait:
            self.all_done.wait()
        self._time_to_stop.set()
        # wake the submitter thread if it's waiting for a job
        self.submit(None)
        if wait:
            self._dispatcher_thread.join()
            self._collector_thread.join()
