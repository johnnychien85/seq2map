function A = dlt(x,y,dim)
% A = DLT(x,y) solves for the projective transformation matrix A with respect to
% the linear system y ~ Ax where ~ denotes equality up to a scale, using the
% Direct Linear Transformation technique. A is a m-by-n matrix, x is a n-by-k
% matrix that contains k source points in column vector form and y is a m-by-k
% matrix containning k target points in column vector form. The solution is
% normalised as any multiple of A also satisfies the equation.
%
% A = DLT(x,y,dim) further specifies the vanishing dimension 0 < dim <= m, by
% which the remaining rows in y are divided. By default dim is set to m, the
% last row of y. The vanashing dimension has to be chosen carefully to avoid
% the singularity of division by zero.
%
% Example:
%
%   A = rand(3,4);
%   x = rand(4,12);
%   A = A / norm(A);
%   A - dlt(x,A*x) % should be a small number
%
% By Hsiang-Jen (Johnny) Chien (jchien@aut.ac.nz) from Centre of Robotics and
% Vision (CeRV), Auckland University of Technology, New Zealand.
%
[n,k ] = size(x);
[m,ky] = size(y);

if nargin < 3, q = m; else q = dim; end

% dimensionality check
assert(k == ky);
assert(q > 0 && q <= m);

% make y inhomogeneous
r = setdiff(1:m,q);
y = bsxfun(@rdivide,y(r,:),y(q,:));

% build the homogeneous linear system
A = zeros(m,n,k,m-1);

for i = 1:numel(r)
    j = r(i);
    b = -bsxfun(@times,x,y(i,:));
    A(j,:,:,i) = reshape(x,1,n,k);
    A(q,:,:,i) = reshape(b,1,n,k);
end

% convert to a big flat matrix
A = reshape(A,m*n,[]);

% solve the homogeneous linear system using SVD
[~,~,V] = svd(A*A');

% the solution minimising |Ax|^2 is the right singular vector
% corresponding to the smallest singular value
A = reshape(V(:,end),m,n);

% some normalisation, optional
A = A / norm(A) * sign(A(1,1));