clc
clear
close all

eps = 1e-10;

tic
poisson = chebfun2(@(x,y) green(x,y),[0 1 0 1],'eps',eps);
toc

tic
[U, S, V] = svd(poisson);
toc

k = 100;
poissonk = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';

% figure;
% plot(poisson)
% colorbar;
% xlabel('x','FontSize',15)
% ylabel('s','FontSize',15)
% % title('G(x,y) for Poisson problem')
% 
% figure;
% plot(poissonk)
% colorbar;
% xlabel('x','FontSize',15)
% ylabel('s','FontSize',15)
% % title(['Rank ',num2str(k),' G(x,y) for Poisson problem'])
% 
% error = poisson-poissonk;
% enorm = norm(error);
% renorm = enorm/norm(poisson);
% disp(enorm);
% disp(renorm);
