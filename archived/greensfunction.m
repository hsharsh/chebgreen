 clc; clear; close all;

% Chebfun
N = 1001;
h = 1 / (N - 1);
x = (0:h:1)';

dom = [0,1];

% Differential operator
N = chebop(@(x,u) -diff(u,2), dom);

% Boundary conditions
N.bc = @(x,u) [u(dom(1)); u(dom(2))];

f = randomf();

u = N \ f;

fx = f(x);
ux = u(x);
plot(x,fx);  k
hold on; plot(x,ux); hold off;
legend('f','u')

function f = randomf()
    sigma = 0.01;
    dom = [0,1];
    domain_length = dom(2) - dom(1);
    K = chebfun2(@(x,y) exp(-(x-y).^2 / (2 * domain_length ^2 * sigma^2)), [dom,dom]);
    L = chol(K,'lower');
    z = randn(rank(L),1);
    f = L * z;
end

% % Discrete version
% N = 1001;
% h = 1 / (N - 1);
% x = (0:h:1)';
% 
% A = 2*eye(N-2) - diag(ones(N-3, 1), 1) - diag(ones(N-3, 1), -1);
% A = A / h^2;
% A = - A;
% 
% f = randomf(x);
% fc = f(2:end-1);
% 
% u = A \ fc;
% u = [0; u; 0];
% 
% plot(x,f);
% hold on; plot(x,u); hold off;
% legend('f','u')
% 
% function f = randomf(x)
%     sigma = 0.01;
%     dom = [0,1];
%     domain_length = dom(2) - dom(1);
%     K = chebfun2(@(x,y) exp(-(x-y).^2 / (2 * domain_length ^2 * sigma^2)), [dom,dom]);
%     L = chol(K,'lower');
%     z = randn(rank(L),1);
%     fc = L * z;
%     f = fc(x);
% end