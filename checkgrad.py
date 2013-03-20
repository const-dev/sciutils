__all__ = ['checkgrad', 'checkgradf']


import sys
import numpy as np
from scipy.optimize import approx_fprime


_epsilon = np.sqrt(np.finfo(float).eps)


def checkgrad(func):
    """Decorator to check the gradient returned from the objective
    evaluation function.

    The function begin decorated should returns both the function
    value as well as its gradient.

    Example
    -------
    @checkgrad
    def func_grad(x):
        f = (3 * x**2 + 2 * x + 1).sum()
        g = 6 * x + 2
        return f, g

    """

    if not __debug__:
        return func

    def _func_and_grad(x, *args):
        ret = func(x, *args)
        if len(ret) == 2:
            def _func(x, *args): return func(x, *args)[0]
            grad = ret[1]
            approx_grad = approx_fprime(x, _func, _epsilon, *args)
            compare_grad(grad, approx_grad)
        return ret
    return _func_and_grad


def checkgradf(func):
    """Decorator that takes the objective evaluation function as an argument
    to check if its gradient matches the result from the gradient function
    being decorated.

    Example
    -------
    def func(x):
        return (3 * x**2 + 2 * x + 1).sum()

    @checkgradf(func)
    def grad(x):
        return 6 * x + 2

    """

    def _checkgrad(grad_func):
        if not __debug__:
            return grad_func
        def _grad(x, *args):
            grad = grad_func(x, *args)
            approx_grad = approx_fprime(x, func, _epsilon, *args)
            compare_grad(grad, approx_grad)
            return grad
        return _grad
    return _checkgrad


def compare_grad(grad, approx_grad):
    err = np.linalg.norm(grad - approx_grad)
    if err > 1e-6:
        print >> sys.stderr, grad
        print >> sys.stderr, approx_grad
        raise Exception('Gradient error: %g' % err)


if __name__ == '__main__':
    try:
        @checkgrad
        def fg(x):
            f_ = (3 * x**2 + 2 * x + 1).sum()
            g_ = 6 * x + 2 + 0.1
            return f_, g_

        fg(np.ones(5))

    except Exception as e:
        print e

    try:
        def f(x):
            f_ = (3 * x**2 + 2 * x + 1).sum()
            return f_

        @checkgradf(f)
        def g(x):
            g_ = 6 * x + 2 + 0.1
            return g_

        g(np.ones(5))

    except Exception as e:
        print e

