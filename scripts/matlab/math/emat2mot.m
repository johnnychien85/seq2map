% M = EMAT2MOT(E,x1,x2) decomposes essential matrix E into rotation and translation
% elements that together form the motion matrix M. Optional normalised point
% correspondences x1 and x2 are given to return only the valid geometric configuration
% out of all the four possible solutions.
function M = emat2mot(E,x1,x2)
	% Fix E to be an ideal essential matrix
	[U,D,V] = svd(E);

	e = (D(1,1) + D(2,2)) / 2;
	D(1,1) = e;
	D(2,2) = e;
	D(3,3) = 0;
	E = U * D * V';

	[U,~,V] = svd(E);

	W = [0 -1 0; 1 0 0; 0 0 1];
	Z = [0 1 0; -1 0 0; 0 0 0];

	% Possible rotation matrices
	R1 = U * W * V';
	R2 = U * W' * V';

	% Force rotations to be proper, i. e. det(R) = 1
	if det(R1) < 0, R1 = -R1; end
	if det(R2) < 0, R2 = -R2; end

	% Translation vector
	Tx = U * Z * U';
	t = [Tx(3, 2), Tx(1, 3), Tx(2, 1)]';

	R = cat(3, R1, R1, R2, R2);
	t = cat(3, t, -t, t, -t);

	M = repmat(eye(4),1,1,4);
	M(1:3,1:4,:) = [R,t];
    
    if nargin > 1
        M = chieralityCheck(M,x1,x2);
    end
end

function [M,bestIdx] = chieralityCheck(M,x1,x2)
    numPosK = 0;
    bestIdx = 0;
    for i = 1 : size(M,3)
        P0 = [eye(3),zeros(3,1)];
        P1 = M(1:3,:,i);
        [xi,~,ki] = triangulatePoints(x1,x2,P0,P1);	
        n = numel(find(ki>0));
        if i == 1 || n > numPosK
            bestIdx = i;
            numPosK = n;
        end
    end
    M = M(:,:,bestIdx);
end
    