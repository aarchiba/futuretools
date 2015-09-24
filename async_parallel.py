#!/usr/bin/env python
# -*- coding: utf-8 -*-

import concurrent.futures
import asyncio
import time
import numpy as np

def deriv(f, d, x, eps):
    xp = np.array(x)
    xp[d] += eps
    xm = np.array(x)
    xm[d] -= eps
    yp, ym = f(xp), f(xm)
    return (yp-ym)/(xp[d]-xm[d])

def grad(f, x, eps):
    return np.array([deriv(f, d, x, eps) for d in range(len(x))])

@asyncio.coroutine
def pmap(f, xs, loop=None):
    fs = [asyncio.async(f(*x), loop=loop) for x in xs]
    yield from asyncio.wait(fs)
    return [f.result() for f in fs]

class ParallelDerivative:
    def __init__(self, f, eps=1e-6):
        self.f = f
        self.eps = eps
        self.loop = asyncio.get_event_loop()
        self.pool = concurrent.futures.ProcessPoolExecutor()

    def gradient(self, x):
        return self.loop.run_until_complete(self.grad_p(x))

    @asyncio.coroutine
    def deriv_p(self, d, x):
        @asyncio.coroutine
        def fwrap(x):
            return (yield from self.loop.run_in_executor(None, self.f, x))
        xp = np.array(x)
        xp[d] += self.eps
        xm = np.array(x)
        xm[d] -= self.eps
        yp, ym = yield from pmap(fwrap, [(xp,), (xm,)], loop=self.loop)
        # Check for error limits goes here; loop as needed
        return (yp-ym)/(xp[d]-xm[d])

    @asyncio.coroutine
    def grad_p(self, x):
        return np.array((yield from pmap(
            self.deriv_p,
            [(d, x) for d in range(len(x))],
            loop=self.loop)))


if __name__=='__main__':
    import time

    v = np.array([1.,2.,3.,4.])
    def f(x):
        print("call for",x)
        time.sleep(1)
        print("return for",x)
        return np.dot(v,x)

    x = np.zeros(len(v))
    eps = 1e-6
    print("sequential")
    print(grad(f,x,eps))
    D = ParallelDerivative(f, eps)
    print("parallel")
    print(D.gradient(x))
