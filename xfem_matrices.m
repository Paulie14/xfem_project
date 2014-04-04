clear;
close all;
diary on;

% ####################################################################################   matrix_1 
disp('SYSTEM matrix:');
SX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1.m');
%[r_s,c_s,sym_s,pd_s] = matrix_prop(SX1);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SX1))));
LSX1 = L*SX1*L';
%[r_s,c_s,sym_s,pd_s] = matrix_prop(LSX1);

disp('----------------------------------------------');

disp('FEM matrix:');
AX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1_a.m');
%[r_a,c_a,sym_a,pd_a] = matrix_prop(AX1);

% PRECONDITIONED FEM MATRIX
disp('preconditioned FEM matrix:');
[m,n] = size(AX1);
LAX1 = LSX1(1:m,1:n);
[r_s,c_s,sym_s,pd_s] = matrix_prop(LAX1);

disp('----------------------------------------------');

disp('Enrichment matrix:');
EX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EX1);

% PRECONDITIONED ENRICHMENT MATRIX
disp('preconditioned Enrichment matrix:');
[me,ne] = size(EX1);
LEX1 = LSX1((m+1):(m+me),(n+1):(n+ne));
[r_s,c_s,sym_s,pd_s] = matrix_prop(LEX1);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');


disp('SYSTEM matrix:');
SXX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1.m');
%[r_s,c_s,sym_s,pd_s] = matrix_prop(SXX1);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SXX1))));
LSXX1 = L*SXX1*L';
%[r_s,c_s,sym_s,pd_s] = matrix_prop(LSXX1);

disp('----------------------------------------------');

disp('FEM matrix:');
AXX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1_a.m');
%[r_a,c_a,sym_a,pd_a] = matrix_prop(AXX1);

% PRECONDITIONED FEM MATRIX
disp('preconditioned FEM matrix:');
[m,n] = size(AXX1);
LAXX1 = LSX1(1:m,1:n);
[r_s,c_s,sym_s,pd_s] = matrix_prop(LAXX1);

disp('----------------------------------------------');

disp('Enrichment matrix:');
EXX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EXX1);

% PRECONDITIONED ENRICHMENT MATRIX
disp('preconditioned Enrichment matrix:');
[me,ne] = size(EXX1);
LEXX1 = LSXX1((m+1):(m+me),(n+1):(n+ne));
[r_s,c_s,sym_s,pd_s] = matrix_prop(LEXX1);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

disp('SYSTEM matrix:');
SS1=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_1.m');
% [r_s,c_s,sym_s,pd_s] = matrix_prop(SS1);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SS1))));
LSS1 = L*SS1*L';
%[r_s,c_s,sym_s,pd_s] = matrix_prop(LSS1);

disp('----------------------------------------------');

disp('FEM matrix:'); AS1=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_1_a.m');
% [r_a,c_a,sym_a,pd_a] = matrix_prop(AS1);

% PRECONDITIONED FEM MATRIX
disp('preconditioned FEM matrix:');
[m,n] = size(AS1);
LAS1 = LSS1(1:m,1:n);
[r_s,c_s,sym_s,pd_s] = matrix_prop(LAS1);
disp('----------------------------------------------');

disp('Enrichment matrix:');
ES1=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_1_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(ES1);

% PRECONDITIONED ENRICHMENT MATRIX
disp('preconditioned Enrichment matrix:');
[me,ne] = size(ES1);
LES1 = LSS1((m+1):(m+me),(n+1):(n+ne));
[r_s,c_s,sym_s,pd_s] = matrix_prop(LES1);
disp('----------------------------------------------');



%{
% ####################################################################################   matrix_1 
disp('FEM matrix:');
AX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1_a.m');
[r_a,c_a,sym_a,pd_a] = matrix_prop(AX1);
disp('----------------------------------------------');

disp('Enrichment matrix:');
EX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EX1);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SX1=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_1.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SX1);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AXX1=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_1_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AXX1);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
EXX1=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_1_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EXX1);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SXX1=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_1.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SXX1);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AS1=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_1_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AS1);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
ES1=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_1_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(ES1);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SS1=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_1.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SS1);

disp('#################################################');
disp('#####################################    MATRIX_1');
disp('#################################################');








% ####################################################################################   matrix_2 
disp('FEM matrix:');
AX2=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_2_a.m');
[r_a,c_a,sym_a,pd_a] = matrix_prop(AX2);
disp('----------------------------------------------');

disp('Enrichment matrix:');
EX2=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_2_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EX2);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SX2=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_2.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SX2);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AXX2=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_2_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AXX2);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
EXX2=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_2_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EXX2);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SXX2=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_2.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SXX2);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AS2=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_2_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AS2);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
ES2=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_2_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(ES2);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SS2=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_2.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SS2);

disp('#################################################');
disp('#####################################    MATRIX_2');
disp('#################################################');







% ####################################################################################   matrix_3 
disp('FEM matrix:');
AX3=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_3_a.m');
[r_a,c_a,sym_a,pd_a] = matrix_prop(AX3);
disp('----------------------------------------------');

disp('Enrichment matrix:');
EX3=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_3_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EX3);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SX3=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_3.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SX3);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AXX3=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_3_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AXX3);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
EXX3=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_3_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EXX3);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SXX3=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_3.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SXX3);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AS3=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_3_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AS3);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
ES3=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_3_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(ES3);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SS3=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_3.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SS3);

disp('#################################################');
disp('#####################################    MATRIX_3');
disp('#################################################');



%}

% ####################################################################################   matrix_4 
disp('SYSTEM matrix:');
SX4=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_4.m');
%[r_s,c_s,sym_s,pd_s] = matrix_prop(SX4);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SX4))));
LSX4 = L*SX4*L';
%[r_s,c_s,sym_s,pd_s] = matrix_prop(LSX4);

disp('----------------------------------------------');

disp('FEM matrix:');
AX4=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_4_a.m');
%[r_a,c_a,sym_a,pd_a] = matrix_prop(AX4);

% PRECONDITIONED FEM MATRIX
disp('preconditioned FEM matrix:');
[m,n] = size(AX4);
LAX4 = LSX4(1:m,1:n);
[r_s,c_s,sym_s,pd_s] = matrix_prop(LAX4);

disp('----------------------------------------------');

disp('Enrichment matrix:');
EX4=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_4_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EX4);

% PRECONDITIONED ENRICHMENT MATRIX
disp('preconditioned Enrichment matrix:');
[me,ne] = size(EX4);
LEX4 = LSX4((m+1):(m+me),(n+1):(n+ne));
[r_s,c_s,sym_s,pd_s] = matrix_prop(LEX4);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');


disp('SYSTEM matrix:');
SXX4=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_4.m');
%[r_s,c_s,sym_s,pd_s] = matrix_prop(SXX4);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SXX4))));
LSXX4 = L*SXX4*L';
%[r_s,c_s,sym_s,pd_s] = matrix_prop(LSXX4);

disp('----------------------------------------------');

disp('FEM matrix:');
AXX4=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_4_a.m');
%[r_a,c_a,sym_a,pd_a] = matrix_prop(AXX4);

% PRECONDITIONED FEM MATRIX
disp('preconditioned FEM matrix:');
[m,n] = size(AXX4);
LAXX4 = LSX4(1:m,1:n);
[r_s,c_s,sym_s,pd_s] = matrix_prop(LAXX4);

disp('----------------------------------------------');

disp('Enrichment matrix:');
EXX4=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_4_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EXX4);

% PRECONDITIONED ENRICHMENT MATRIX
disp('preconditioned Enrichment matrix:');
[me,ne] = size(EXX4);
LEXX4 = LSXX4((m+1):(m+me),(n+1):(n+ne));
[r_s,c_s,sym_s,pd_s] = matrix_prop(LEXX4);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

disp('SYSTEM matrix:');
SS4=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_4.m');
% [r_s,c_s,sym_s,pd_s] = matrix_prop(SS4);

disp('preconditioned SYSTEM matrix:');
% PRECONDITIONED SYSTEM MATRIX
L=sqrt(inv(diag(diag(SS4))));
LSS4 = L*SS4*L';
%[r_s,c_s,sym_s,pd_s] = matrix_prop(LSX4);

disp('----------------------------------------------');

disp('FEM matrix:'); AS4=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_4_a.m');
% [r_a,c_a,sym_a,pd_a] = matrix_prop(AS4);

% PRECONDITIONED FEM MATRIX
disp('preconditioned FEM matrix:');
[m,n] = size(AS4);
LAS4 = LSS4(1:m,1:n);
[r_s,c_s,sym_s,pd_s] = matrix_prop(LAS4);
disp('----------------------------------------------');

disp('Enrichment matrix:');
ES4=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_4_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(ES4);

% PRECONDITIONED ENRICHMENT MATRIX
disp('preconditioned Enrichment matrix:');
[me,ne] = size(ES4);
LES4 = LSS4((m+1):(m+me),(n+1):(n+ne));
[r_s,c_s,sym_s,pd_s] = matrix_prop(LES4);
disp('----------------------------------------------');



disp('#################################################');
disp('#####################################    MATRIX_4');
disp('#################################################');





%{

% ####################################################################################   matrix_5 
%  disp('FEM matrix:');
%  AX5=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_5_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AX5);
%  disp('----------------------------------------------');
%  
%  disp('Enrichment matrix:');
%  EX5=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_5_e.m');
%  [r_e,c_e,sym_e,pd_e] = matrix_prop(EX5);
%  disp('----------------------------------------------');
%  
%  disp('SYSTEM matrix:');
%  SX5=dlmread('./001/output/square_convergence_xfem_ramp_model_bc/matrix_5.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SX5);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

disp('FEM matrix:');
AXX5=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_5_a.m');
[r_a,c_a,sym_a,pd_a] = matrix_prop(AXX5);
disp('----------------------------------------------');

disp('Enrichment matrix:');
EXX5=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_5_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(EXX5);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SXX5=dlmread('./001/output/square_convergence_xfem_shift_model_bc/matrix_5.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SXX5);


disp('----------------------------------------------###');
disp('----------------------------------------------###');
disp('----------------------------------------------###');

%  disp('FEM matrix:');
%  AS5=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_5_a.m');
%  [r_a,c_a,sym_a,pd_a] = matrix_prop(AS5);
%  disp('----------------------------------------------');

disp('Enrichment matrix:');
ES5=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_5_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(ES5);
disp('----------------------------------------------');

%  disp('SYSTEM matrix:');
%  SS5=dlmread('./001/output/square_convergence_sgfem_model_bc/matrix_5.m');
%  [r_s,c_s,sym_s,pd_s] = matrix_prop(SS5);

%}
diary off;