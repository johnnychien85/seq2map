% I0 = IMWARPDP(It,D01) warps image I1 to I0 via disparity map D01 following
% I0(x,y) = I1(x-D01(x,y),y).
%
function I0 = imwarpdp(I1,D01,c0,c1,cj)
    [m,n,k] = size(I1);
	assert(size(D01,1) == m && size(D01,2) == n);

    if nargin > 2
        assert(nargin == 5);
        rectified = checkCanonical(c0,c1);
    else
        rectified = true;
    end

    I0 = zeros(m,n,k,'like',I1);

    if rectified
        for y = 1 : m
            % profile of disparity image in y-th row
            f = D01(y,:);

            % find pixels with valid disparities and their corresponding pixels in I1
            dst = find(f > 0 & isfinite(f)); % I0
            src = dst - f(dst);              % I1

            % exclude out-of-range pixels
            idx = find(src >= 1 & src <= n);
            src = src(idx);
            dst = dst(idx);

            % interpolation requires at least two data points
            if numel(idx) < 2, continue; end

            % apply interpolation for each colour channel
            for d = 1 : k
                I0(y,dst,d) = interp1(1:n,double(I1(y,:,d)),src);
            end
        end
        return;
    end

    x3d = dp2pts(D01,c0,cj);
    x2d = reshape(pts2image(reshape(x3d,[],3),c1),m,n,2);

    idx = find(x2d(:,:,1) >= 0 & x2d(:,:,1) <= n & ...
               x2d(:,:,2) >= 0 & x2d(:,:,2) <= m);

    x = x2d(:,:,1);
    y = x2d(:,:,2);

    for d = 1 : k
        I0d = I0(:,:,d);
        I1d = I1(:,:,d);
        I0d(idx) = interp2(I1d(:,:,d),x(idx)+1,y(idx)+1); % plus 1 as INTERP2 uses 1-based subscripts by default
        I0(:,:,d) = I0d;
    end
end

function canonical = checkCanonical(c0,c1)
    E = c1.E * invmot(c0.E);
    R = E(1:3,1:3);
    t = E(1:3,4);

    canonical = all(reshape(R == eye(3),1,[])) & t(2) == 0 & t(3) == 0;
end