clc
clear
close all
prefx = chebfunpref();
prefy = chebfunpref();
techx       = prefx.tech();
techy       = prefy.tech();
tprefx      = techx.techPref;
tprefy      = techy.techPref;
minSample   = [tprefx.minSamples, tprefy.minSamples];

maxSample   = [tprefx.maxLength, tprefy.maxLength];
sampleTest  = prefx.cheb2Prefs.sampleTest | prefy.cheb2Prefs.sampleTest;
maxRank     = [prefx.cheb2Prefs.maxRank, prefy.cheb2Prefs.maxRank];
pseudoLevel = min(prefx.cheb2Prefs.chebfun2eps, prefy.cheb2Prefs.chebfun2eps);
dom = [0,1, 0 1];
vectorize = true;
isTrig = [false false];
op = @green;

% minSample needs to be a power of 2 when building periodic CHEBFUN2 objects or
% a ones plus power of 2 otherwise.  See #1771.
minSample( isTrig) = 2.^(floor(log2(minSample(isTrig))));
minSample(~isTrig) = 2.^(floor(log2(minSample(~isTrig) - 1))) + 1;

factor  = 4; % Ratio between size of matrix and no. pivots.
isHappy = 0; % If we are currently unresolved.
failure = 0; % Reached max discretization size without being happy.

grid = minSample;

% Sample function on a Chebyshev tensor grid:
[xx, yy] = points2D(grid(1), grid(2), dom, prefx, prefy);
size(xx)
vals = evaluate(op, xx, yy, false);

% Does the function blow up or evaluate to NaN?:
vscale = max(abs(vals(:)));
if ( isinf(vscale) )
    error('CHEBFUN:CHEBFUN2:constructor:inf', ...
        'Function returned INF when evaluated');
elseif ( any(isnan(vals(:)) ) )
    error('CHEBFUN:CHEBFUN2:constructor:nan', ...
        'Function returned NaN when evaluated');
end

% Two-dimensional version of CHEBFUN's tolerance:
[relTol, absTol] = getTol(xx, yy, vals, dom, pseudoLevel);
prefx.chebfuneps = relTol;
prefy.chebfuneps = relTol;

%% %%% PHASE 1: %%%
% Do GE with complete pivoting:
[pivotVal, pivotPos, rowVals, colVals, iFail] = completeACA(vals, absTol, factor);

strike = 1;
% grid <= 4*(maxRank-1)+1, see Chebfun2 paper.
while ( iFail && all(grid <= factor*(maxRank-1)+1) && strike < 3)
    % Refine sampling on tensor grid:
    grid(1) = gridRefine(grid(1), prefx);
    grid(2) = gridRefine(grid(2), prefy);
    [xx, yy] = points2D(grid(1), grid(2), dom, prefx, prefy);
    vals = evaluate(op, xx, yy, vectorize); % resample
    vscale = max(abs(vals(:)));
    % New tolerance:
    [relTol, absTol] = getTol(xx, yy, vals, dom, pseudoLevel);
    prefx.chebfuneps = relTol;
    prefy.chebfuneps = relTol;
    % New GE:
    [pivotVal, pivotPos, rowVals, colVals, iFail] = ...
                             completeACA(vals, absTol, factor);
    % If the function is 0+noise then stop after three strikes.
    if ( abs(pivotVal(1)) < 1e4*vscale*relTol )
        strike = strike + 1;
    end
    disp(pivotVal(1:5))
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = myPoints(n, dom, pref)
% Get the sample points that correspond to the right grid for a particular
% technology.

% What tech am I based on?:
tech = pref.tech();

if ( isa(tech, 'chebtech2') )
    x = chebpts( n, dom, 2 );   % x grid.
elseif ( isa(tech, 'chebtech1') )
    x = chebpts( n, dom, 1 );   % x grid.
elseif ( isa(tech, 'trigtech') )
    x = trigpts( n, dom );   % x grid.
else
    error('CHEBFUN:CHEBFUN2:constructor:mypoints:techType', ...
        'Unrecognized technology');
end

end

function [grid, nesting] = gridRefine( grid, pref )
% Hard code grid refinement strategy for tech.

% What tech am I based on?:
tech = pref.tech();

% What is the next grid size?
if ( isa(tech, 'chebtech2') )
    % Double sampling on tensor grid:
    grid = 2^( floor( log2( grid ) ) + 1) + 1;
    nesting = 1:2:grid;
elseif ( isa(tech, 'trigtech') )
    % Double sampling on tensor grid:
    grid = 2^( floor( log2( grid ) + 1 ));
    nesting = 1:2:grid;
elseif ( isa(tech, 'chebtech1' ) )
    grid = 3 * grid;
    nesting = 2:3:grid;
else
    error('CHEBFUN:CHEBFUN2:constructor:gridRefine:techType', ...
        'Technology is unrecognized.');
end

end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xx, yy] = points2D(m, n, dom, prefx, prefy)
% Get the sample points that correspond to the right grid for a particular
% technology.

if ( nargin == 4 )
    prefy = prefx;
end

% What tech am I based on?:
techx = prefx.tech();
techy = prefy.tech();

if ( isa(techx, 'chebtech2') )
    x = chebpts( m, dom(1:2), 2 );
elseif ( isa(techx, 'chebtech1') )
    x = chebpts( m, dom(1:2), 1 );
elseif ( isa(techx, 'trigtech') )
    x = trigpts( m, dom(1:2) );
else
    error('CHEBFUN:CHEBFUN2:constructor:points2D:tecType', ...
        'Unrecognized technology');
end

if ( isa(techy, 'chebtech2') )
    y = chebpts( n, dom(3:4), 2 );
elseif ( isa(techy, 'chebtech1') )
    y = chebpts( n, dom(3:4), 1 );
elseif ( isa(techy, 'trigtech') )
    y = trigpts( n, dom(3:4) );
else
    error('CHEBFUN:CHEBFUN2:constructor:points2D:tecType', ...
        'Unrecognized technology');
end

[xx, yy] = meshgrid( x, y );

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vals = evaluate( op, xx, yy, flag )
% EVALUATE  Wrap the function handle in a FOR loop if the vectorize flag is
% turned on.

if ( flag )
    vals = zeros( size( yy, 1), size( xx, 2) );
    for jj = 1 : size( yy, 1)
        for kk = 1 : size( xx , 2 )
            vals(jj, kk) = op( xx( 1, kk) , yy( jj, 1 ) );
        end
    end
else
    vals = op( xx, yy );  % Matrix of values at cheb2 pts.
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [relTol, absTol] = getTol(xx, yy, vals, dom, pseudoLevel)
% GETTOL     Calculate a tolerance for the Chebfun2 constructor.
%
%  This is the 2D analogue of the tolerance employed in the chebtech
%  constructors. It is based on a finite difference approximation to the
%  gradient, the size of the approximation domain, the internal working
%  tolerance, and an arbitrary (2/3) exponent.

[m, n] = size( vals );
grid = max( m, n );
dfdx = 0;
dfdy = 0;
if ( m > 1 && n > 1 )
    % Remove some edge values so that df_dx and df_dy have the same size.
    dfdx = diff(vals(1:m-1,:),1,2) ./ diff(xx(1:m-1,:),1,2); % xx diffs column-wise.
    dfdy = diff(vals(:,1:n-1),1,1) ./ diff(yy(:,1:n-1),1,1); % yy diffs row-wise.
elseif ( m > 1 && n == 1 )
    % Constant in x-direction
    dfdy = diff(vals,1,1) ./ diff(yy,1,1);
elseif ( m == 1 && n > 1 )
    % Constant in y-direction
    dfdx = diff(vals,1,2) ./ diff(xx,1,2);
end
% An approximation for the norm of the gradient over the whole domain.
Jac_norm = max( max( abs(dfdx(:)), abs(dfdy(:)) ) );
vscale = max( abs( vals(:) ) );
relTol = grid.^(2/3) * pseudoLevel; % this should be vscale and hscale invariant
absTol = max( abs(dom(:) ) ) * max( Jac_norm, vscale) * relTol;

end

function [pivotValue, pivotElement, rows, cols, ifail] = ...
    completeACA(A, absTol, factor)
% Adaptive Cross Approximation with complete pivoting. This command is
% the continuous analogue of Gaussian elimination with complete pivoting.
% Here, we attempt to adaptively find the numerical rank of the function.

% Set up output variables.
[nx, ny] = size(A);
width = min(nx, ny);        % Use to tell us how many pivots we can take.
pivotValue = zeros(1);      % Store an unknown number of Pivot values.
pivotElement = zeros(1, 2); % Store (j,k) entries of pivot location.
ifail = 1;                  % Assume we fail.

% Main algorithm
zRows = 0;                  % count number of zero cols/rows.
[infNorm, ind] = max( abs ( reshape(A,numel(A),1) ) );
[row, col] = myind2sub( size(A) , ind);

% Bias toward diagonal for square matrices (see reasoning below):
if ( ( nx == ny ) && ( max( abs( diag( A ) ) ) - infNorm ) > -absTol )
    [infNorm, ind] = max( abs ( diag( A ) ) );
    row = ind;
    col = ind;
end

scl = infNorm;

% The function is the zero function.
if ( scl == 0 )
    % Let's pass back the zero matrix that is the same size as A. 
    % This ensures that chebfun2( zeros(5) ) has a 5x5 (zero) coefficient 
    % matrix.  
    pivotValue = 0;
    rows = zeros(1, size(A,2));
    cols = zeros(size(A,1), 1);
    ifail = 0;
else
    rows(1,:) = zeros(1, size(A, 2));
    cols(:,1) = zeros(size(A, 1), 1);
end

while ( ( infNorm > absTol ) && ( zRows < width / factor) ...
        && ( zRows < min(nx, ny) ) )
    rows(zRows+1,:) = A(row,:);
    cols(:,zRows+1) = A(:,col);              % Extract the columns.
    PivVal = A(row,col);
    A = A - cols(:,zRows+1)*(rows(zRows+1,:)./PivVal); % One step of GE.
    
    % Keep track of progress.
    zRows = zRows + 1;                       % One more row is zero.
    pivotValue(zRows) = PivVal;              % Store pivot value.
    pivotElement(zRows,:)=[row col];         % Store pivot location.
    
    % Next pivot.
    [ infNorm , ind ] = max( abs ( A(:) ) ); % Slightly faster.
    [ row , col ] = myind2sub( size(A) , ind );
    
    % Have a bias towards the diagonal of A, so that it can be used as a test
    % for nonnegative definite functions. (Complete GE and Cholesky are the
    % same as nonnegative definite functions have an absolute maximum on the
    % diagonal, except there is the possibility of a tie with an off-diagonal
    % absolute maximum. Bias toward diagonal maxima to prevent this.)
    if ( ( nx == ny ) && ( max( abs( diag( A ) ) ) - infNorm ) > -absTol )
        [infNorm, ind] = max( abs ( diag( A ) ) );
        row = ind;
        col = ind;
    end
end
if ( infNorm <= absTol )
    ifail = 0;                               % We didn't fail.
end
if ( zRows >= (width/factor) )
    ifail = 1;                               % We did fail.
end

end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [row, col] = myind2sub(siz, ndx)
% My version of ind2sub. In2sub is slow because it has a varargout. Since this
% is at the very inner part of the constructor and slowing things down we will
% make our own. This version is about 1000 times faster than MATLAB ind2sub.

vi = rem( ndx - 1, siz(1) ) + 1 ;
col = ( ndx - vi ) / siz(1) + 1;
row = ( vi - 1 ) + 1;

end
