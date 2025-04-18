% This code generates the datasets for the examples written in the folder
% "examples".
% To generate the dataset corresponding to the file helmholtz.m,
% run the command
% generate_gl_example('helmholtz',theta);

function generate_example(example_name, Nsample, lambda, Nf, Nu, noise_level, seed, theta, varargin)
    % Add warning about Chebfun
    assert(exist('chebfun') == 2,"This code requires the Chebfun package. See http://www.chebfun.org/download/ for installation details.")

    rng(seed);

    % Load the differential operator example
    if nargin > 7
        disp(['### Example = ', example_name, ' @ theta = ',num2str(theta), ' ###'])
        output_example = feval(example_name, theta);
    else
        disp(['### Example = ', example_name, ' ###'])
        output_example = feval(example_name);
    end

    disp(['Number of samples: ',num2str(Nsample)]);
    disp(['Length scale: ',num2str(lambda)]);
    disp(['Nf: ',num2str(Nf)]);
    disp(['Nu: ',num2str(Nu)]);
    disp(['Noise: ',num2str(noise_level*100),'%']);
    disp('---------------------------------------');

    diff_op = output_example{1};
    dom = diff_op.domain;
    
    i = 2;
    linear = true;
    while i <= length(output_example)
        % Get covariance kernel frequency
        if strcmp(output_example{i},"lambda") && i< length(output_example)
            lambda = output_example{i+1};
            i = i+1;
        % Get exact Green's function
        elseif strcmp(output_example{i},"ExactGreen") && i< length(output_example)
            ExactGreen = output_example{i+1};
            i = i+1;
        % Don't rescale if nonlinear operator
        elseif strcmp(output_example{i},"nonlinear")
            linear = false;
        end
        i = i+1;
    end
    
    % Reshape domain
    dom = [dom(1), dom(end)];
    
    % Number of input-output functions
    if linear
        size_system = size(diff_op.linop);
    else
        size_system = [diff_op.nargin-1,diff_op.nargin-1];
    end
    n_input = size_system(2);
    n_output = size_system(1);

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
    sigma = domain_length*lambda;
    if strcmp(diff_op.bc,'periodic')
        K = chebfun2(@(x,y)exp(-2*sin(pi*abs(x-y)/domain_length).^2/sigma^2), [dom,dom], 'trig');
    else
        K = chebfun2(@(x,y)exp(-(x-y).^2/(2*sigma^2)), [dom,dom]);
    end
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
        for k = 2:n_input
            % Sample from a Gaussian process
            f = generate_random_fun(L);
            rhs = [rhs;f];
        end

        % Solve the equation
        u = solvebvp(diff_op, rhs, options);
        
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

    disp(['Forcing functions represented by ', num2str(length(f)), ' chebyshev points.'])

    % Compute homogeneous solution
    u_hom = solvebvp(diff_op, 0, options);
    
    % Convert to chebfun
    if isa(u_hom,'chebmatrix')
        u_hom_col = [];
        for n = 1:length(u_hom.blocks)
            u_hom_col = [u_hom_col, u_hom.blocks{n}];
        end
        u_hom = u_hom_col;
    end
    
    U_hom = u_hom(X);
    
    % Normalize the solution for homogeneous problems
    if all(iszero(u_hom)) && linear
        scale = max(abs(U),[],'all');
        U = U/scale;
        F = F/scale;
    end
    
    % Add Gaussian noise to the solution
    U = U + (noise_level*randn(size(U)) .* mean(abs(U)));

    % Save the data
    formatSpec = '%.2f';
    if nargin > 8
        savePath = sprintf('datasets/%s/%s-%s',example_name, num2str(theta,formatSpec), varargin{1});
    elseif nargin > 7
        savePath = sprintf('datasets/%s/%s',example_name,num2str(theta,formatSpec));
    else
        savePath = sprintf('datasets/%s/%s',example_name);
    end
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    if exist("ExactGreen")
        save(sprintf('%s/data.mat',savePath),"X","Y","U","F","U_hom","XG","YG","ExactGreen")
    else
        save(sprintf('%s/data.mat',savePath),"X","Y","U","F","U_hom","XG","YG")
    end
    
    % Plot the training data
    plot_data = false;
    if nargin > 9 && varargin{1}
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