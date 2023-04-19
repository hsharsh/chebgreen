import chebpy
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod
from .quasimatrix import Quasimatrix

class Chebfun2(ABC):
    def __init__(self,g = None, domain = np.array([-1, 1, -1, 1]), prefx = chebpy.core.settings.DefaultPreferences, prefy = None, vectorize = False):

        # Default domain is [-1,1] x [-1, 1]
        if type(domain) is not np.ndarray:
            self.domain = np.array(domain)
        else:
            self.domain = domain
        self.cols, self.rows, self.pivotValues, self.pivotLocations = self.constructor(g, prefx, prefy, vectorize)

        self.rank = len(self.pivotValues)
        self.vscale = 0
        self.cornervalues = np.array([0,0,0,0])

        # Write a sampletest here
    
    def __repr__(self):
        header = f"chebfun2 object\n"
        toprow = "     domain       rank     corner values\n"
        rowdta = (f"[{self.domain[0]},{self.domain[1]}] x [{self.domain[2]},{self.domain[3]}]     {self.rank}       "
            f"[{self.cornervalues[0]} {self.cornervalues[1]} {self.cornervalues[2]} {self.cornervalues[3]}]\n")
        btmrow = f"vertical scale = {self.vscale}"
        return header + toprow + rowdta + btmrow
    

    def constructor(self, g = None, prefx = chebpy.core.settings.DefaultPreferences, prefy = None, vectorize = False):

        # Define this somewhere in a config file
        minSample = np.array([17,17])
        maxSample = np.array(np.power(2,[prefx.maxpow2,prefy.maxpow2]))
        maxRank = np.array([513,513])

        if prefy == None:
            prefy = prefx
        
        assert (prefx.tech == 'Chebtech2' and prefy.tech == 'Chebtech2'), "CHEBFUN:CHEBFUN2:constructor, Unrecognized technology"
        
        pseudoLevel = min(prefx.eps, prefy.eps)
        
        factor = 4.0        # Ration between the size of matrix and #pivots.
        isHappy = 0         # If we are currently unresolved.
        failure = 0         # Reached max discretization size without being happy.

        minSample = np.power(2,np.floor(np.log2(minSample-1)))+1


        while not isHappy and not failure:
            # Remove this
            # break
            grid = minSample

            # Sample function on a Chebyshev tensor grid:
            xx, yy = points2D(grid[0],grid[1],self.domain,prefx,prefy)
            vals = evaluate(g, xx, yy, vectorize)
            
            # Does the function blow up or evaluate to nan?:
            vscale = np.max(np.abs(vals[:]))

            if vscale == np.inf:
                raise RuntimeError('Function returned INF when evaluated')
            elif (vals[:] == np.nan).any():
                raise RuntimeError('Function returned NaN when evaluated')
            

            relTol, absTol = getTol(xx, yy, vals, self.domain, pseudoLevel)
            prefx.eps = relTol
            prefy.eps = relTol

            #### Phase 1:
            # Do GE with complete pivoting:
            pivotVal, pivotPos, rowVals, colVals, iFail = completeACA(vals, absTol, factor)

            strike = 1
            while iFail and (grid <= factor*(maxRank-1)+1).any() and strike < 3:
                # Refine sampling on tensor grid:
                grid[0], _ = gridRefine(grid[0], prefx)
                grid[1], _ = gridRefine(grid[1], prefy)
                xx, yy = points2D(grid[0],grid[1],self.domain,prefx,prefy)
                vals = evaluate(g, xx, yy) # Resample
                vscale = np.max(np.abs(vals[:]))
                
                #New Tolerance
                relTol, absTol = getTol(xx, yy, vals, self.domain, pseudoLevel)
                prefx.eps = relTol
                prefy.eps = relTol
                
                pivotVal, pivotPos, rowVals, colVals, iFail = completeACA(vals, absTol, factor)
                
                # If the function is 0+noise then stop after three strikes.
                if np.abs(pivotVal[0]) < 1e4*vscale*relTol:
                    strike += 1
            
            # If the rank of the function is above maxRank then stop.
            if (grid > factor*(maxRank-1)+1).any():
                failure = 1
                raise RuntimeWarning('CHEBFUN:CHEBFUN2:constructor:rank, Not a low-rank function.')
            
            # Check if the column and row slices are resolved. Hardcoded for Chebtech2
            colTech = chebpy.core.chebtech.Chebtech2.initvalues(values = np.sum(colVals,axis = 1), interval = self.domain[2:])
            resolvedCols = chebpy.core.algorithms.happinessCheck(tech = colTech, vals = colVals, pref = prefy)
            rowTech = chebpy.core.chebtech.Chebtech2.initvalues(values =  np.sum(rowVals.T, axis = 1), interval = self.domain[:2])
            resolvedRows = chebpy.core.algorithms.happinessCheck(tech = rowTech, vals = rowVals.T, pref = prefx)
            isHappy = resolvedRows and resolvedCols

            if len(pivotVal) == 1 and pivotVal == 0:
                pivPos = np.array([[0,0]])
                isHappy = 1
            else:
                pivPos = np.array([[xx[0,pivotPos[j,1]], yy[pivotPos[j,0],0]] for j in range(len(pivotVal))])
                PP = pivotPos
            
            #### Phase 2:
            # Resolve along the column and row slices:
            n, m = grid[1], grid[0]

            while not isHappy and not failure:
                if not resolvedCols:
                    n, nesting = gridRefine(n, prefy)
                    xx, yy = np.meshgrid(pivPos[:,0], chebpy.chebpts(n, self.domain[2:], prefy)[0])
                    colVals = evaluate(g, xx, yy, vectorize)
                    PP[:,0] = nesting[PP[:,0]]
                else:
                    xx, yy = np.meshgrid(pivPos[:,0], chebpy.chebpts(n, self.domain[2:], prefy)[0])
                    colVals = evaluate(g, xx, yy, vectorize)
                
                if not resolvedRows:
                    m, nesting = gridRefine(m, prefy)
                    xx, yy = np.meshgrid(chebpy.chebpts(m, self.domain[:2], prefx)[0], pivPos[:,1])
                    rowVals = evaluate(g, xx, yy, vectorize)
                    PP[:,1] = nesting[PP[:,1]]
                else:
                    xx, yy = np.meshgrid(chebpy.chebpts(m, self.domain[:2], prefx)[0], pivPos[:,1])
                    rowVals = evaluate(g, xx, yy, vectorize)

                nn = len(pivotVal)

                for kk in range(nn):
                    colVals[:, kk+1:] = colVals[:, kk+1:] - colVals[:,kk] @ (rowVals[kk, PP[kk+1:nn,1]]/pivotVal[kk])
                    rowVals[kk+1:, :] = rowVals[kk+1:, :] - colVals[PP[kk+1:nn,0], kk] @ (rowVals[kk,:]/pivotVal[kk])

                # !!! Check if this is correct
                if nn == 1:
                    rowVals = rowVals.T

                # Are the columns and rows resolved now?
                if not resolvedCols:
                    colTech = chebpy.core.chebtech.Chebtech2.initvalues(values = np.sum(colVals,axis = 1), interval = self.domain[2:])
                    resolvedCols = chebpy.core.algorithms.happinessCheck(tech = colTech, vals = colVals, pref = prefy)

                if not resolvedRows:
                    rowTech = chebpy.core.chebtech.Chebtech2.initvalues(values =  np.sum(rowVals.T, axis = 1), interval = self.domain[:2])
                    resolvedRows = chebpy.core.algorithms.happinessCheck(tech = rowTech, vals = rowVals.T, pref = prefx)

                isHappy = resolvedRows and resolvedCols

                # Stop if degree is over maxLength
                sampleCheck = [m,n] < maxSample
                if not sampleCheck.all():
                    raise RuntimeWarning(f'CHEBFUN:CHEBFUN2:constructor:notResolved, Unresolved with maximum CHEBFUN length:{maxSample[np.argwhere(~sampleCheck)[0,0]]}')
                
        # !! Check if this is needed/correct
        if np.linalg.norm(colVals) == 0 or np.linalg.norm(rowVals) == 0:
            colVals = 0
            rowVals = 0
            pivotVal = np.inf
            pivotPos = np.array([0,0])
            isHappy = 1

        cols = Quasimatrix(data = chebpy.chebfun(colVals, domain = np.array(self.domain[2:]), prefs = prefy), transposed = False) 
        rows = Quasimatrix(data = chebpy.chebfun(rowVals.T, domain = np.array(self.domain[:2]), prefs = prefx), transposed = True)
        pivotValues = pivotVal
        pivotLocations = pivotPos

        # Write a Sample Test
        
        return cols, rows, pivotValues, pivotLocations
    
    # --------------------
    #  operator overloads
    # --------------------
    def __add__(self, f):
        raise NotImplementedError

    def __getitem__(self, key):
        """
        Evaluate a learned chebfun2 object on numeric, array, or meshgrid inputs
        """
        x, y = key
        if (isinstance(x,slice) and x == slice(None)) and (isinstance(y,slice) and y == slice(None)):
            return self
        
        C,D,R = self.cdr()

        if (isinstance(x,slice) and x == slice(None)):
            if (isinstance(y,int) or isinstance(y,float)):
                y = np.array([y])
            if y.dtype == np.int64 or y.dtype == np.float64:
                # Make evaluation points a vector.
                y = y.reshape(-1)
                return C[y,:] @ D * R
        
        if (isinstance(y,slice) and y == slice(None)):
            if (isinstance(x,int) or isinstance(x,float)):
                x = np.array([x])
            if x.dtype == np.int64 or x.dtype == np.float64:
                # Make evaluation points a vector.
                x = x.reshape(-1)
                return C * (D @ R[:,x])

        eps = np.finfo(np.float64).eps
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and min(x.shape) > 1 and x.shape == y.shape and len(x.shape) == 2:
            if np.max(x - x[0:1,:]) <= 10*eps and np.max(y - y[:,0:1]) < 10*eps:
                x, y = x[0,:], y[:,0]
                return C[y,:] @ D @ R[:,x]
            elif np.max(y - y[0:1,:]) <= 10*eps and np.max(x - x[:,0:1]) < 10*eps:
                x, y = x[:,0], y[0,:]
                return (C[y,:] @ D @ R[:,x]).T
            else:
                # Evaluate at matrices, but they are not from meshgrid:
                m, n = x.shape

                # Unroll the loop that is the longest
                if m > n:
                    out = np.array([np.sum(C[y[:,i],:] @ D * R[:,x[:,i]].T, axis = 1) for i in range(n)]).T
                else:
                    out = np.array([np.sum(C[y[j,:],:] @ D * R[:,x[j,:]].T, axis = 1) for j in range(m)])
        if (isinstance(x,int) or isinstance(x,float)) and (isinstance(y,int) or isinstance(y,float)):
            x, y = np.array([x]), np.array([y])
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and x.shape == y.shape and (len(x.shape) == 1 or min(x.shape) == 1):
            shape = x.shape
            x, y = x.reshape(-1), y.reshape(-1)
            return (C[y,:] * R[:,x].T @ np.diag(D)).reshape(shape)
        
        raise NotImplementedError('Cannot evaluate chebfun2 object with given inputs')
        
    def cdr(self):
        return self.cols, np.diag(1/self.pivotValues), self.rows

def Max(A):
    return np.max(A), np.argmax(A)

def points2D(m, n, dom, prefx, prefy):
    # Get the sample points that correspond to the right grid for a particular
    # technology.
    
    if prefy == None:
        prefy = prefx
        
    # What tech am I based on?
    techx, techy = prefx.tech, prefy.tech
    
    if techx == "Chebtech2":
        x = chebpy.chebpts(m, dom[:2])
    else:
        raise Exception('CHEBFUN:CHEBFUN2:constructor:points2D:tecType, Unrecognized technology')

    if techy == "Chebtech2":
        y = chebpy.chebpts(n, dom[2:])
    else:
        raise Exception('CHEBFUN:CHEBFUN2:constructor:points2D:tecType, Unrecognized technology')

    [xx, yy] = np.meshgrid(x,y)
    return xx, yy



def evaluate(op, xx, yy, vectorize = 0):
    # EVALUATE  Wrap the function handle in a FOR loop if the vectorize flag is
    # turned on.

    if(vectorize):
        vals = np.zeros((yy.shape[0], xx.shape[1]))
        for jj in range(yy.shape[0]):
            for kk in range(xx.shape[1]):
                vals[jj, kk] = op( xx(0,kk), yy(jj,0))
    else:
        vals = op(xx,yy)  # Matrix of values at cheb2 pts.
    return vals

def getTol(xx, yy, vals, dom, pseudoLevel):
    # GETTOL     Calculate a tolerance for the Chebfun2 constructor.
    #
    #  This is the 2D analogue of the tolerance employed in the chebtech
    #  constructors. It is based on a finite difference approximation to the
    #  gradient, the size of the approximation domain, the internal working
    #  tolerance, and an arbitrary (2/3) exponent.

    m, n = vals.shape
    grid = max(m,n)
    dfdx, dfdy = 0, 0

    if m > 1 and n > 1:
        # Remove some edge values so that df_dx and df_dy have the same size.
        dfdx = np.diff(vals[:m-1,:],1,1) / np.diff(xx[:m-1,:],1,1) # xx diffs column-wise.
        dfdy = np.diff(vals[:,:n-1],1,0) / np.diff(yy[:,:n-1],1,0) # yy diffs row-wise.
    elif m > 1 and n == 1:
        # Constant in x-direction
        dfdy = np.diff(vals,1,1) / np.diff(yy,1,1)
    elif m == 1 and n > 1:
        # Constant in y-direction
        dfdx = np.diff(vals,1,2) / np.diff(xx,1,2)

    # An approximation for the norm of the gradient over the whole domain.
    Jac_norm = max(np.max(np.abs(dfdx[:])), np.max(np.abs(dfdy[:])))
    vscale = np.max(np.abs(vals[:]))
    relTol = grid**(2/3) * pseudoLevel # This should be vscale and hscale invariant
    absTol = np.max(np.abs(dom[:])) * max(Jac_norm, vscale) * relTol

    return relTol, absTol

def gridRefine( grid, pref):
    # Hard code grid refinement strategy

    # What tech am I based on?:
    tech = pref.tech

    # What is the next grid size?
    if tech == "Chebtech2":
        # Double sampling on tensor grid:
        grid = np.power(2,np.floor(np.log2(grid))+1) + 1
        nesting = np.arange(1,grid+1,2)
    else:
        raise RuntimeError('CHEBFUN:CHEBFUN2:constructor:gridRefine:techType, Technology is unrecognized.')
    return grid, nesting

def completeACA(A, absTol, factor):
    # Adaptive Cross Approximation with complete pivoting. This command is
    # the continuous analogue of Gaussian elimination with complete pivoting.
    # Here, we attempt to adaptively find the numerical rank of the function.

    # Set up output variables.
    nx, ny = A.shape
    width = min(nx, ny)        # Use to tell us how many pivots we can take.
    pivotValue = np.empty(int(np.ceil(width/factor)))         # Store an unknown number of Pivot values.
    pivotElement = np.empty((int(np.ceil(width/factor)),2), dtype = np.int64) # Store (j,k) entries of pivot location.
    pivotValue[:], pivotElement[:] = np.nan, -1
    ifail = 1                  # Assume we fail.

    # Main algorithm
    zRows = 0                  # count number of zero cols/rows.
    infNorm, ind = Max(np.abs(A))
    
    # Check here for errors bc python's default between row and column major is different.
    row, col = np.unravel_index(ind, A.shape) 
    # Error possiblity

    # Bias toward diagonal for square matrices (see reasoning below):
    if (nx == ny) and (np.max(np.abs(np.diag(A))) - infNorm) > -absTol:
        infNorm, ind = Max(np.abs(np.diag(A)))
        row = ind
        col = ind

    scl = infNorm

    # The function is the zero function.
    if scl == 0:
        # Let's pass back the zero matrix that is the same size as A. 
        # This ensures that chebpy2(np.zeros(5)) has a 5x5 (zero) coefficient 
        # matrix.  
        pivotValue = 0
        rows = np.zeros((1, A.shape[1]))
        cols = np.zeros((A.shape[0], 1))
        ifail = 0
    else:
        rows = np.zeros((1, A.shape[1]))
        cols = np.zeros((A.shape[0], 1))
    
    while (infNorm > absTol) and (zRows < width/factor) and (zRows < min(nx,ny)):
        # Check if zRows+1 is correct here
        if zRows == 0:
            rows = A[row,:].reshape((1,-1))
            cols = A[:,col].reshape((-1,1))
        else:
            rows = np.vstack([rows,A[row,:].reshape((1,-1))])
            cols = np.hstack([cols,A[:,col].reshape((-1,1))])            # Extract the columns.
        PivVal = A[row,col]
        A = A - cols[:,zRows].reshape(-1,1) @ (rows[zRows,:].reshape(1,-1)/PivVal) # One step of GE.

        # Keep track of progress.
        pivotValue[zRows] = PivVal                 #pivotValue[zRows] = PivVal             # Store pivot value.
        pivotElement[zRows,:] = [row, col]   #pivotElement[zRows,:]=[row col]        # Store pivot location.
        zRows = zRows + 1                         # One more row is zero.

        # Next pivot.
        infNorm , ind = Max(np.abs(A)) # Slightly faster.
        row , col = np.unravel_index(ind, A.shape) # Check for col/row major

        # Have a bias towards the diagonal of A, so that it can be used as a test
        # for nonnegative definite functions. (Complete GE and Cholesky are the
        # same as nonnegative definite functions have an absolute maximum on the
        # diagonal, except there is the possibility of a tie with an off-diagonal
        # absolute maximum. Bias toward diagonal maxima to prevent this.)
        if (nx == ny) and ((np.max(np.abs(np.diag(A))) - infNorm) > -absTol):
            infNorm, ind = Max(np.abs(np.diag(A)))
            row = ind
            col = ind
    
    # print(infNorm)
    
    if infNorm <= absTol:
        ifail = 0                               # We didn't fail.

    if zRows >= (width/factor):
        ifail = 1                               # We did fail.
    
    return pivotValue, pivotElement, rows, cols, ifail