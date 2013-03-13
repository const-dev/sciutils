import sys
import numpy as np
from scipy.optimize import approx_fprime


_epsilon = np.sqrt(np.finfo(float).eps)


def checkgrad(func):
    """Decorator to check the gradient returned from the objective
    evaluation function

    Parameters
    ----------
    func: callable func(x, *args)
          Function whose returned derivative is to be checked
          func returns both the function value and its gradient:
          f, g = func(x, *args)
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
    """Decorator to check the gradient that takes the objective evaluation
    function as an argument

    Parameters
    ----------
    func: callable func(x, *args)
          Objective evaluation function corresponds to the gradient function
    """

    def _checkgrad(grad_func):
        """
        grad_func: callable grad_func(x, *args)
                   Function whose returned derivative is to be checked
        """

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

