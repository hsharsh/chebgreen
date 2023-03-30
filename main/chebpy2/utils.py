import chebpy as cp
import numpy as np

def scaleNodes(x, dom):
    # SCALENODES   Scale the Chebyshev nodes X from [-1,1] to DOM.
    if dom[0] == -1 and dom[1] == 1:
        # Nodes are already on [-1, 1]
        return x

    # Scale the nodes:
    return dom[2]*(x + 1)/2 + dom[1]*(1 - x)/2

def chebpts(n, dom = cp.core.settings.DefaultPreferences.domain, type = 2):
    #chebpts    Chebyshev points.
    #   chebpts(N) returns N Chebyshev points of the 2nd-kind in [-1,1].
    #
    #   chebpts(N, D), where D is vector of length 2 and N is a scalar integer,
    #   scales the nodes and weights for the interval [D(1),D(2)].

    ##################################################################################
    #   [Mathematical reference]:
    #   Jarg Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
    #   quadrature rules", BIT Numerical Mathematics, 46, (2006), pp 195-202.
    ##################################################################################

    # Create a dummy CHEBTECH of appropriate type to access static methods.
    if type == 1:
        f = cp.core.chebtech.Chebtech
    elif type == 2:
        f = cp.core.chebtech.Chebtech2
    else:
        raise Exception('CHEBFUN:chebpts:type, Unknown point type.') 

    if len(n) == 1:         # Single grid
        # Call the static CHEBTECH.CHEBPTS() method:
        x = f._chebpts(n)
        # Scale the domain:
        x = scaleNodes(x, dom)
    else:                  # Piecewise grid.
        raise NotImplementedError
    return x