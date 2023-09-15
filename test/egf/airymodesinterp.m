clc
clear
close all

% Load the modes of an Airy problem at different parameter learned using
% randomized SVD
load('airymodes.mat');

C1 = [];
C2 = [];
C3 = [];
Cr = [];

s = 3;

% Convert the eigenvector matrices into eigenfunction quasimatrices
% s = size(U1,2);
for i = 1:s
    C1 = [C1 chebfun(U1(:,i),[0,1],'trig')];
    C2 = [C2 chebfun(U2(:,i),[0,1],'trig')];
    C3 = [C3 chebfun(U3(:,i),[0,1],'trig')];
    Cr = [Cr chebfun(U(:,i),[0,1],'trig')];
end

x = 0:1/2000:1;

% Interpolate the quasimatrices

% Lift to tangent space
C0 = C1;
T1 = C1 - C0 * (C0'*C1 + C1'*C0)/2;
T2 = C2 - C0 * (C0'*C2 + C2'*C0)/2;
T3 = C3 - C0 * (C0'*C3 + C3'*C0)/2;

% Interpolate with lagrange polynomials
% T = -0.1666667 * T1 + 0.9 * T2 + 0.26666667 * T3;
T = T3;

% Retract from tangent space
[C,] = qr(C0 + T);

% Plot the interpolated model at the target parameter against a target
% model at the same parameter
% j = 1; % Mode index
% hold on;
% plot(C1(:,j));
% plot(C2(:,j));
% plot(C3(:,j));
% plot(C(:,j));
% plot(Cr(:,j));
% legend('Interpolant 1','Interpolant 2','Interpolant 3','Interpolation','Target');


U1 = U1(:,1:s);
U2 = U2(:,1:s);
U3 = U3(:,1:s);
U = U(:,1:s);

U1 = U1/sqrt(2000);
U3 = U3/sqrt(2000);
P = U3 - U1 * (U1'*U3 + U3' * U1)/2;
[Cd,R] = qr(U1 + P,"econ");

j = 1; % Mode index
hold on;
plot(C3(:,j),'b');
plot(C(:,j),'g');
plot(0:1/2000:1,Cd(:,j)*sqrt(2000),'--r');
legend('True','Cont. Interp','Disc. Interp');
