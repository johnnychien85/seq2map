% MOT2FMAT converts pose matrix M = [R,t] to fundamental matrix F
% with respect to cameras K1 and K2. For corresponding image points
% (x1,x2) it holds x2'*F*x1 = 0. Optionally the epipoles are returned
% by the second and third outputs.
%
% See also mot2emat, invcam
%
function [F,e1,e2] = mot2fmat(M,K1,K2)
    if nargin < 3, K2 = K1; end
    F = invcam(K2)' * mot2emat(M) * invcam(K1);
    
    if nargout > 1, e1 = homo2eucl(M(1:3,4)'*M(1:3,1:3)*K1'); end
    if nargout > 2, e2 = homo2eucl(M(1:3,4)'           *K2'); end
end