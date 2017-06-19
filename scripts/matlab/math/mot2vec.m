% convert post matrix to a 6-vector
function x = mot2vec(M,angleAxis)
    if nargin < 2, angleAxis = false; end
    R = M(1:3,1:3);
    t = M(1:3,4);
    if angleAxis
        r = vrrotmat2vec(R)';
        r = norm(r(1:3)) * r(1:3);
    else
        [r1,r2,r3] = dcm2angle(R);
        r = rad2deg([r1,r2,r3])';
    end
	x = [r;t];
end