function [E,inliers,err] = estimateEssentialMatrix(x0,x1,K0,K1,epsilon,confidence,iter)
	K0_inv = inv(K0);
	K1_inv = inv(K1);

	vec2emat = @(x) mot2emat(vec2mot(x));

	[E,err,inliers] = ransac(...
		size(x0,1),	...
		8,			...
		@(idx) pts2emat(x0(idx,:),x1(idx,:)), ...
		@(E)   sqrt(computeEpipolarError(x0,x1,K1_inv'*E*K0_inv,'sampson')), ...
		epsilon,	...
		confidence, ...
		iter,       ...
		0.2         ...
	);
	
	function E = pts2emat(x0,x1)
		F = estimateFundamentalMatrix(x0,x1,'Method','Norm8Point');
		E = K1'*F*K0;
		M = chieralityCheck(emat2mot(E),x0,x1);

		% nonlinear adjustment
		opts = optimoptions('lsqnonlin','Display','none','MaxIter',10);
		x = lsqnonlin(@(x)computeEpipolarError(x0,x1,K1_inv'*vec2emat(x)*K0_inv,'sampson'),mot2vec(M),[],[],opts);
		E = vec2emat(x);
	end
	
	function [M,bestIdx] = chieralityCheck(M,x0,x1)
		numPosK = 0;
		bestIdx = 0;
		
		for i = 1 : size(M,3)
			P0 = K0 * [eye(3),zeros(3,1)];
			P1 = K1 * M(1:3,:,i);
			[xi,~,ki] = triangulatePoints(x0,x1,P0,P1);	
			n = numel(find(ki>0));
			if i == 1 || n > numPosK
				bestIdx = i;
				numPosK = n;
			end
		end
		M = M(:,:,bestIdx);
	end
end

