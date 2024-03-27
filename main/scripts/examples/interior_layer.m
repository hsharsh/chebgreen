function output_example = interior_layer(theta)
    % Interior layer equation
    
    % Define the domain.
    dom = [0,1];
    
    
    % Differential operator
    N = chebop(@(x,u)theta*diff(u,2)+x*(x^2-0.5)*diff(u)+3*(x^2-0.5)*u, dom);
    
    % Boundary conditions
    N.bc = @(x,u) [u(dom(1)); u(dom(2))];
    
    % Output
    output_example = {N};
    end