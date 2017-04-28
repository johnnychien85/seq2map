% ESTIMATEESSENTIALMATRIX finds an essential matrix from 2D point correspondences given
% camera matrices. The outliers are rejected using RANSAC.
function [E,inliers,err] = estimateEssentialMatrix(x1,x2,K1,K2,epsilon,confidence,iter)
	K1_inv = inv(K1);
	K2_inv = inv(K2);

	vec2emat = @(x) mot2emat(vec2mot(x));

	[E,err,inliers] = ransac(...
		size(x1,1),	...
		8,			...
		@(idx) pts2emat(x1(idx,:),x2(idx,:)), ...
		@(E)   sqrt(computeEpipolarError(x1,x2,K2_inv'*E*K1_inv,'sampson')), ...
		epsilon,	...
		confidence, ...
		iter,       ...
		0.2         ...
	);
	
    % find an essential matrix given exactly 8 image correspondences
	function E = pts2emat(x1,x2)
		F = estimateFundamentalMatrix(x1,x2,'Method','Norm8Point');
		E = K2'*F*K1;
		M = emat2mot(E,eucl2cano(x1,K1),eucl2cano(x2,K2));

		% nonlinear adjustment
		opts = optimoptions('lsqnonlin','Display','none','MaxIter',10);
		x = lsqnonlin(@(x)computeEpipolarError(x1,x2,K2_inv'*vec2emat(x)*K1_inv,'sampson'),mot2vec(M),[],[],opts);
		E = vec2emat(x);
	end
end

