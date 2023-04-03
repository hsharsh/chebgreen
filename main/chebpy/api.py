"""User-facing functions"""

import numpy as np

from .core.bndfun import Bndfun
from .core.chebfun import Chebfun
from .core.utilities import Domain
from .core.settings import _preferences


def chebfun(f = None, domain = _preferences.domain, n = None, pref = _preferences):
    """Chebfun constructor"""
    # chebfun()
    _preferences = pref
    if f is None:
        cf = Chebfun.initempty()
        _preferences.reset()
        return cf

    # chebfun(lambda x: f(x), ... )
    if hasattr(f, "__call__"):
        cf = Chebfun.initfun(f, domain, n)
        _preferences.reset()
        return cf

    # chebfun('x', ... )
    if isinstance(f, str) and len(f) == 1 and f.isalpha():
        if n:
            cf = Chebfun.initfun(lambda x: x, domain, n)
            _preferences.reset()
            return cf
        else:
            cf = Chebfun.initidentity(domain)
            _preferences.reset()
            return cf

    if isinstance(f, np.ndarray) and len(f.shape) == 2:
        if f.shape[1] == 1:
            cf = Chebfun.initvalues(f, domain)
            _preferences.reset()
            return cf
        else:
            cf = np.array([Chebfun.initvalues(f[:,i], domain) for i in range(f.shape[1])]).reshape((1,-1))
            _preferences.reset()
            return cf
    try:
        # chebfun(3.14, ... ), chebfun('3.14', ... )
        cf = Chebfun.initconst(float(f), domain)
        _preferences.reset()
        return cf
    except (OverflowError, ValueError):
        raise ValueError(f"Unable to construct const function from {{{f}}}")


def pwc(domain=[-1, 0, 1], values=[0, 1], pref = _preferences):
    """Initialise a piecewise-constant Chebfun"""
    _preferences = pref
    funs = []
    intervals = [x for x in Domain(domain).intervals]
    for interval, value in zip(intervals, values):
        funs.append(Bndfun.initconst(value, interval))
    cf = Chebfun(funs)
    _preferences.reset()
    return cf
