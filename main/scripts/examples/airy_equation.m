function output_example = airy_equation(theta)
% Parameterized airy equation

% Define the domain.
dom = [0,1];

% Differential operator
N = chebop(@(x,u) diff(u,2)-theta^2*x*u, dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

% Output
output_example = {N};
end