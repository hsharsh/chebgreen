function output_example = advection_diffusion(theta)
    % Mean one condition: http://www.chebfun.org/examples/ode-linear/NonstandardBCs.html
    
    % Define the domain.
    dom = [-1,1];
    
    % Differential operator
    N = chebop(@(x,u) diff(u,2) + theta*diff(u,1) + u, dom);
    
    % Boundary conditions
    N.bc = @(x,u) [u(dom(1)); u(dom(2))];
    
    % Output
    output_example = {N};
    end