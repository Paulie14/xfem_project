clear;
close all;

%A=dlmread('./001/output/xmodel_simple_convergence/xfem_matrix.m');	
%A=dlmread('./001/output/model_simple_convergence/fem_matrix.m');
%A=dlmread('./001/backup/xmodel_simple_convergence/xfem_matrix.m');	


disp('FEM matrix:');
A=dlmread('./001/backup/xmodel_simple_convergence/matrix_0_a.m');
%A=dlmread('./001/backup/xmodel_simple_convergence_no_shift/matrix_0_a.m');
[r_a,c_a,sym_a,pd_a] = matrix_prop(A);
disp('----------------------------------------------');

disp('Enrichment matrix:');
E=dlmread('./001/backup/xmodel_simple_convergence/matrix_0_e.m');
%E=dlmread('./001/backup/xmodel_simple_convergence_no_shift/matrix_0_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(E);
disp('----------------------------------------------');

disp('SYSTEM matrix:');
S=dlmread('./001/backup/xmodel_simple_convergence/matrix_0.m');
%S=dlmread('./001/backup/xmodel_simple_convergence_no_shift/matrix_0.m');
[r_s,c_s,sym_s,pd_s] = matrix_prop(S);


disp('----------------------------------------------');
disp('----------------------------------------------');
disp('----------------------------------------------');
%}

disp('FEM matrix:');
A=dlmread('./001/backup/xmodel_simple_convergence_shift_wrong2/matrix_0_a.m');
[r_a,c_a,sym_a,pd_a] = matrix_prop(A);
disp('----------------------------------------------');

disp('Enrichment matrix:');
E=dlmread('./001/backup/xmodel_simple_convergence_shift_wrong2/matrix_0_e.m');
[r_e,c_e,sym_e,pd_e] = matrix_prop(E);
disp('----------------------------------------------');

disp('SYSTEM matrix:');
S=dlmread('./001/backup/xmodel_simple_convergence_shift_wrong2/matrix_0.m');
[r_s,c_s,sym_s,pd_s] = matrix_prop(S);
disp('----------------------------------------------');