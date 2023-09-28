function generate_fractional(example_name, Nsample, lambda, Nf, Nu, noise_level, theta, varargin)
    % Add warning about Chebfun
    assert(exist('chebfun') == 2,"This code requires the Chebfun package. See http://www.chebfun.org/download/ for installation details.")

    disp(['Number of samples: ',num2str(Nsample)]);
    disp(['Length scale: ',num2str(lambda)]);
    disp(['Nf: ',num2str(Nf)]);
    disp(['Nu: ',num2str(Nu)]);
    disp(['Noise: ',num2str(noise_level*100),'%']);
    disp('---------------------------------------');

    dom = [-1,1];
    
    n_input = 1;
    n_output = 1;

    % Training points for f
    Y = linspace(dom(1), dom(2), Nf)';
    
    % Training points for u
    X = linspace(dom(1), dom(2), Nu)';

    % Evaluation points for G
    NGx = 1000;
    NGy = 1000;
    XG = linspace(dom(1), dom(2), NGx)';
    YG = linspace(dom(1), dom(2), NGy)';
    
    % Define the Gaussian process kernel
    domain_length = dom(end) - dom(1);
    K = chebfun2(@(x,y)exp(-2*sin(pi*abs(x-y)/domain_length).^2/lambda^2), [dom,dom], 'trig');

    % Compute the Cholesky factorization of K
    L = chol(K, 'lower');
    
    % Setup preferences for solving the problem.
    options = solver_options();

    % Initialize data arrays
    U = zeros(Nu, Nsample, n_output);
    F = zeros(Nf, Nsample, n_input);

    % Loop over the number of sampled functions f
    for i = 1:Nsample
        disp(['Step = ',num2str(i),'/',num2str(Nsample)])
        
        % Sample from a Gaussian process
        f = generate_random_fun(L);
        rhs = f;

        % Solve the equation
        n = length(f);
        % Ensure n is odd
        if mod(n,2) == 0
            n = n+1;
        end
        
        % Define differentiation matrix of -d^(2s)/d^2s on [-1,1]
        Dx2 = (-1*trigspec.diffmat(n,2)).^theta;
        % Add zero mean condition
        Dx2(floor(n/2)+1,floor(n/2)+1) = 1;
        
        % Solve equation
        fx = trigcoeffs(f, n);
        uc = Dx2 \ fx;
        
        % Convert Fourier coeffs to values
        u = trigtech.coeffs2vals(uc);
        
        % Make chebfun
        u = chebfun(u, f.domain, 'trig');

        % Convert to chebfun
        if isa(u,'chebmatrix')
            u_col = [];
            for n = 1:length(u.blocks)
                u_col = [u_col, u.blocks{n}];
            end
            u = u_col;
        end
        if isa(rhs,'chebmatrix')
            rhs_col = [];
            for n = 1:length(rhs.blocks)
                rhs_col = [rhs_col, rhs.blocks{n}];
            end
            rhs = rhs_col;
        end
        
        % Evaluate at the training points
        U(:,i,:) = u(X);
        F(:,i,:) = rhs(Y);
    end
    
    % Compute homogeneous solution
    f = chebfun(@(x) 0*x, dom, 'trig');
    n = length(f);

    % Define differentiation matrix of -d^(2s)/d^2s on [-1,1]
    Dx2 = (-1*trigspec.diffmat(n,2)).^theta;
    % Add zero mean condition
    Dx2(floor(n/2)+1,floor(n/2)+1) = 1;
    
    % Solve equation
    fx = trigcoeffs(f, n);
    uc = Dx2 \ fx;
    
    % Convert Fourier coeffs to values
    u_hom = trigtech.coeffs2vals(uc);
    
    % Make chebfun
    u_hom = chebfun(u_hom, f.domain, 'trig');
    
    % Convert to chebfun
    if isa(u_hom,'chebmatrix')
        u_hom_col = [];
        for n = 1:length(u_hom.blocks)
            u_hom_col = [u_hom_col, u_hom.blocks{n}];
        end
        u_hom = u_hom_col;
    end
    
    U_hom = u_hom(X);
    
    % Add Gaussian noise to the solution
    U = U.*(1 + noise_level*randn(size(U)));

    % Save the data
    formatSpec = '%.2f';
    savePath = sprintf('datasets/%s',example_name);
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    if nargin > 6
        if exist("ExactGreen")
            save(sprintf('%s/%s.mat',savePath,num2str(theta,formatSpec)),"X","Y","U","F","U_hom","XG","YG","ExactGreen")
        else
            save(sprintf('%s/%s.mat',savePath,num2str(theta,formatSpec)),"X","Y","U","F","U_hom","XG","YG")
        end
    else
        if exist("ExactGreen")
            save(sprintf('%s/data.mat',savePath),"X","Y","U","F","U_hom","XG","YG","ExactGreen")
        else
            save(sprintf('%s/data.mat',savePath),"X","Y","U","F","U_hom","XG","YG")
        end
    end
    
    % Plot the training data
    plot_data = false;
    if nargin > 7 && varargin{1}
        plot_data = true;
    end
    if plot_data
        subplot(1,2,1)
        plot(Y, F)
        xlabel("y")
        title("Training functions f")
        xlim([min(Y),max(Y)])
        axis square

        subplot(1,2,2)
        plot(X, U)
        xlabel("x")
        title("Training solutions u")
        xlim([min(X),max(X)])
        axis square
    end
    
end

function f = generate_random_fun(L)
% Take a cholesky factor L of a covariance kernel and return a smooth
% random function.

% Generate a vector of random numbers
u = randn(rank(L),1);
f = L*u;
end

function options = solver_options()
% Setup preferences for solving the problem.
% Create a CHEBOPPREF object for passing preferences.
% (See 'help cheboppref' for more possible options.)
options = cheboppref();

options.display = 'off'; % 'iter' : Print information at every Newton step

% Option for tolerance.
options.bvpTol = 5e-13;

% Option for damping.
options.damping = false;

% Specify the discretization to use. Possible options are:
%  'values' (default)
%  'coeffs'
%  A function handle (see 'help cheboppref' for details).
options.discretization = 'values';

end