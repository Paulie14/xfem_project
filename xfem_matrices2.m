clear all;
clc;
close all;
diary on;

% ####################################################################################   matrix_1 
disp('SYSTEM matrix:');
SX1=dlmread('./002/esco/square_convergence_xfem_shift_model_04/matrix_1.m');
%[r_s,c_s,sym_s,pd_s] = matrix_prop(SX1);
m1 = size(SX1,1);
SX1 = SX1(1:m1-1,1:m1-1); 

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SX1))));
LSX1 = L*SX1*L';
[r_s,c_s,sym_s,pd_s] = matrix_prop(LSX1);

disp('----------------------------------------------');

% ####################################################################################   matrix_1 
disp('SYSTEM matrix:');
SX2=dlmread('./002/esco/square_convergence_sgfem_model_04/matrix_1.m');
%[r_s,c_s,sym_s,pd_s] = matrix_prop(SX1);
m2 = size(SX2,1);
SX2 = SX2(1:m2-1,1:m2-1);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SX2))));
LSX2 = L*SX2*L';
[r_s,c_s,sym_s,pd_s] = matrix_prop(LSX2);

disp('----------------------------------------------');