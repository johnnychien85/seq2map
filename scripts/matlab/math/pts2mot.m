function M = pts2mot(x1,x2,K)
    E = estimateEssentialMatrix(x1,x2,K,K);
    M = emat2mot(E,eucl2cano(x1,K),eucl2cano(x2,K));
end

%{
function [M,E] = pts2mot(x1,x2,K)
	F = estimateFundamentalMatrix(x0,x1);
	E = K'*F*K;
	M = chieralityCheck(emat2mot(E),x0,x1,K);
end

function M = chieralityCheck(M,x0,x1,K)
	numPosK = 0;
	bestIdx = 0;
	
	for i = 1 : size(M,3)
		P0 = K * [eye(3),zeros(3,1)];
		P1 = K * M(1:3,:,i);
		[xi,~,ki] = triangulatePoints(x0,x1,P0,P1);	
		n = numel(find(ki>0));
		if i == 1 || n > numPosK
			bestIdx = i;
			numPosK = n;
		end
	end
	M = M(:,:,bestIdx);
end
%}