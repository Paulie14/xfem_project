clear;


%A=dlmread('./001/output/xmodel_simple_convergence/xfem_matrix.m');	
A=dlmread('./001/output/model_simple_convergence/fem_matrix.m');

disp('dimensions and rank: ');
[m,n] = size(A)
rank = rank(A,eps)


disp('Octave matrix type: ');
matrix_type(A)

disp('Symetry (=norm(A-AT)): ');
norm(A-A')

%computation of eigenvalues and condition number
disp('Condition number:');
lambda = eig(A);
max_lambda = max(lambda)
min_lambda = min(lambda)
cond_ = abs( max_lambda/min_lambda)
cond (A)
condest(A)

disp('Positive definite:');
pos_def = 0;
for i=1:n
    temp =  (det( A(1:i, 1:i) ) > 0)
    pos_def = pos_def + temp;
end
if(pos_def == rank) 
	disp('YES');
else
	disp('NO');
endif


%diagonally dominant
%diagA = diag( A )' ;    % Get the diagonal entries of S
%sum( abs( A - diag( diagA ) ) );  % subtract off the diagonal entries, take the absolute value and sum
%diag( A )' > sum( abs( A - diag(diag(A)) ) );