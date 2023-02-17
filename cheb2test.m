poisson = chebfun2(@(x,y) green(x,y),[0 1 0 1],'splitting','on');
[U, S, V] = svd(poisson);

k = 50;
poissonk = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';