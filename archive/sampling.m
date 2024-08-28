function error = sampling(N,sigma)
    a = 0; b = 1;
    X = a:1/(N-1):b;

    plotfunction = true;

    
    dom = [min(X), max(X)];
    domain_length = dom(end) - dom(1);
    
    K = chebfun2(@(x,y) exp(-(x-y).^2 / (2 * domain_length ^2 * sigma^2)), [dom, dom]);
    L = chol(K,'lower');
    
    f = generate_random_function(L);
    disp(f)
    z = (chebpts(N)+1)*((b-a)/2) + a;
    fu = f(X);
    fc = interp1(X,fu,z);
    g = chebfun(fc,[a,b]);
    % g = chebfun(@(x) interp1(X,fu,x), [a,b]);
    disp(f)
    error = norm(f-g)/norm(f);
    
    if plotfunction
        fig = figure;
        fig.Position = [50 50 1600 900];
        hold on;
        plot(f,'r','LineWidth',1.2)
        plot(g,'bx--','LineWidth',1.2)
        % plot(g,'bx','LineWidth',1.2)
        title(['Error = ',num2str(error)],fontsize = 18)
        lgd = legend('Sample','Approximation');
        fontsize(lgd, 16,'points')
    end
end
function f = generate_random_function(L)
    u = randn(rank(L),1);
    f = L * u;
end