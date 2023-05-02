import chebpy
import numpy as np

def scaleNodes(x, dom):
    # SCALENODES   Scale the Chebyshev nodes X from [-1,1] to DOM.
    if dom[0] == -1 and dom[1] == 1:
        # Nodes are already on [-1, 1]
        return x

    # Scale the nodes:
    return dom[2]*(x + 1)/2 + dom[1]*(1 - x)/2

def chebpts(n, dom = chebpy.core.settings.DefaultPreferences.domain, type = 2):
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
        f = chebpy.core.chebtech.Chebtech
    elif type == 2:
        f = chebpy.core.chebtech.Chebtech2
    else:
        raise Exception('CHEBFUN:chebpts:type, Unknown point type.') 

    if isinstance(n,int) or isinstance(n,np.int64) or len(n) == 1:         # Single grid
        # Call the static CHEBTECH.CHEBPTS() method:
        x = f._chebpts(n)
        # Scale the domain:
        x = scaleNodes(x, dom)
    else:                  # Piecewise grid.
        raise NotImplementedError
    return x

def legpoly(n, dom = chebpy.core.settings.DefaultPreferences.domain, normalize = False, prefs = chebpy.core.settings._preferences):
    """
    P = legpoly(N) computes a CHEBFUN of the Legendre polynomial of degree N on
    the interval [-1,1]. N can be a vector of integers, in which case the output
    is an array-valued CHEBFUN.

    P = legpoly(N, D) computes the Legendre polynomials as above, but on the
    interval given by the domain D, which must be bounded.

    P = LEGPOLY(N, D, True) normalises so that
    integral(P(:,j).*P(:,k)) = delta_{j,k}.

    For N <= 1000 LEGPOLY uses the standard recurrence relation.

    There is no current implementation of leg2cheb, or computation using
    weighted QR factorisation of a 2*(N+1) x 2*(N+1) Chebyshev Vandermonde
    matrix, or corresponding to legoply chebfun:
    https://github.com/chebfun/chebfun/blob/master/legpoly.m
    """

    nMax = max(n)
    u, indices = np.unique(n, return_index = True)
    P = np.zeros((nMax+1, max(n.shape)))
    x = chebpts(nMax+1)
    Lk_2, Lk_1 = np.ones((nMax+1,)), x           # P0 and P1, Initial condn for recurrence
    ind = 0
    for k in range(2,nMax+2):
        if u[ind] == k-2:
            if normalize:
                invnrm = np.sqrt((2*k - 3)/np.diff(dom))
                P[:,ind] = Lk_2*invnrm
            else:
                P[:,ind] = Lk_2
            
            # Update recurrence
            temp = Lk_1
            Lk_1 = (2 - 1/k)*x*Lk_1 - (1 - 1/k)*Lk_2
            Lk_2 = temp
            ind += 1
    return chebpy.chebfun(P[:,indices], dom, prefs = prefs)
