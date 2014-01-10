function [r,c,sym,pd] = matrix_prop(A)

disp('dimensions and rank: ');
[m,n] = size(A)
r = rank(A,eps)


disp('Octave matrix type: ');
pd_temp = matrix_type(A)
if(pd_temp == 'Positive Definite')
	pd = 1;
else pd=0;
endif

disp('Symetry (=norm(A-AT)): ');
norm_sym = norm(A-A');
if (norm_sym == 0)
	disp('YES');
	sym = 1;
else
	disp('NO');
	sym = 0;
endif

%computation of eigenvalues and condition number
disp('Condition number:');
lambda = eig(A);
max_lambda = max(lambda)
min_lambda = min(lambda)
c = abs( max_lambda/min_lambda)
cond (A)
condest(A)

%{
disp('Positive definite:');

pos_def = 0;
for i=1:n
    temp =  (det( A(1:i, 1:i) ) > 0);
    pos_def = pos_def + temp;
end
if(pos_def == r) 
	disp('YES');
	pd=1;
else
	disp('NO');
	pd=0;
endif

%}

% diagonally dominant
% diagA = diag( A )' ;    % Get the diagonal entries of S
% sum( abs( A - diag( diagA ) ) );  % subtract off the diagonal entries, take the absolute value and sum
% diag( A )' > sum( abs( A - diag(diag(A)) ) );