#!/usr/bin/env python
# -*- coding: utf-8 -*-

import concurrent.futures

class IpyFuture(concurrent.futures.Future):
    def __init__(self, async_result)
        concurrent.futures.Future.__init__(self)
        self.async_result = async_result

class IpyExecutor(concurrent.futures.Executor):
    def __init__(self, client):
        concurrent.futures.Executor.__init__(self)
        self.client = client
        self.view = self.client.load_balanced_view()

    def submit(self, fn, *args, **kwargs):
        result = self.view.apply(f, args, kwargs,
                                 block=False)
        f = concurrent.futures.Future()
        
