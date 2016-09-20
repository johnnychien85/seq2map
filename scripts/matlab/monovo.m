% MONOVO presents an implementation of monocular visual odometry using
% feature tracking, motion from essential matrix decomposition,
% motion by solving perspective-n-point and particle filtering.
function [M,egomo,stats,pc] = monovo(seq,cam,alpha)
	includes;

	if nargin < 2, cam = 1;     end
	if nargin < 3, alpha = 0.9; end

	cam = seq.cam(cam);

	if ~isfield(cam,'FeatureFiles') || isempty(cam.FeatureFiles)
		hint(seq,cam);
		error 'missing image feature files';
	end

	epsilon = 0.1;
	confidence = 99.5;%99.95;
	hasMotion = ~isempty(cam.M);

	t0 = 1;
	tn = numel(cam.ImageFiles);
	frames = tn - t0;
	uid = 0;
	stats.te   = zeros(frames,1);
	stats.nepi = zeros(frames,1);
	stats.npnp = zeros(frames,1);
	stats.nraw = zeros(frames,1);
	stats.naug = zeros(frames,1);
	stats.nout = zeros(frames,1);
	stats.matchTime = 0;
	stats.filterTime = 0;
	
	% initialise motion matrix stack
    M = repmat(eye(4),1,1,tn);

	K = cam.K;
	K_inv = inv(cam.K);
	
	% initialise features' 3D coordinates of the first frame
	Ii = loadImage(cam,t0);
	Fi = loadFeatures(cam,t0);
	Fi.G = zeros(Fi.NumOfFeatures,3);
	Fi.W = zeros(Fi.NumOfFeatures,1);
	Fi.U = zeros(Fi.NumOfFeatures,1,'uint32');

	% draw something
	%whitebg(figure); a = gca; axis equal; grid on; hold on;
	%cmap = hsv(tn-t0+1);
	
	fprintf('Solving monocular egomotion..\n');
    for t = t0 : tn - 1
		% consider frames ti -> tj
        ti = t;
        tj = t + 1;

		% load features of frame tj
		Ij = loadImage(cam,tj);
		Fj = loadFeatures(cam,tj);
		Fj.G = zeros(Fj.NumOfFeatures,3);
		Fj.W = zeros(Fj.NumOfFeatures,1);
		Fj.U = zeros(Fj.NumOfFeatures,1,'uint32');

		% find matched features
		%idx = matchGoodFeatures(Fi,Fj,K,[],[],epsilon,confidence);
		[idx,Fj,matchTime,filterTime,nraw,naug,nout] = matchGoodFeatures(Fi,Fj,K,Ii,Ij,epsilon,confidence);
		
		stats.matchTime  = stats.matchTime  + matchTime;
		stats.filterTime = stats.filterTime + filterTime;
		stats.nraw(ti-t0+1) = nraw;
		stats.naug(ti-t0+1) = naug;
		stats.nout(ti-t0+1) = nout;

		if numel(idx) < 10
			warning 'feature tracking failed';
			return;
		end
		
		% ID allocation and propogation
		[Fi.U(idx(:,1)),uid] = allocateNewIds(Fi.U(idx(:,1)),uid);
		Fj.U(idx(:,2)) = Fi.U(idx(:,1));

		% ego-motion estimation problem
		egomo(ti).K     = cam.K;
		egomo(ti).K_inv = inv(cam.K);
		
		% epipolar conds
		egomo(ti).epi.src = double(Fi.KP(idx(:,1),:));
		egomo(ti).epi.dst = double(Fj.KP(idx(:,2),:));
		egomo(ti).epi.w   = Fi.W (idx(:,1),:);
		egomo(ti).epi.uid = Fi.U (idx(:,1),:);
		
		% reprojection conds
		Ipnp = idx(find(Fi.G(idx(:,1),3) > 0),:);
		egomo(ti).pnp.src = Fi.G (Ipnp(:,1),:);
		egomo(ti).pnp.dst = double(Fj.KP(Ipnp(:,2),:));
		egomo(ti).pnp.w   = Fi.W (Ipnp(:,1),:);
		egomo(ti).pnp.uid = Fi.U (Ipnp(:,1),:);

		if numel(Ipnp) > 3
			[Mij,inliers] = estimateMotion(egomo(ti).pnp.src,egomo(ti).pnp.dst,K,epsilon*10,confidence,1000);
			outliers = find(~inliers);
			inliers  = find(inliers);
			
			Fi.W(Ipnp(outliers,1)) = 0;
			
			fprintf('%d outliers out of %d rejected from PnP set\n',numel(outliers),size(Ipnp,1));

			Ipnp = Ipnp(inliers,:);
			egomo(ti).pnp.src = egomo(ti).pnp.src(inliers,:);
			egomo(ti).pnp.dst = egomo(ti).pnp.dst(inliers,:);
			egomo(ti).pnp.w   = egomo(ti).pnp.w  (inliers,:);
			egomo(ti).pnp.uid = egomo(ti).pnp.uid(inliers,:);
			
			if nargout > 3
				pc_i = pointCloud(...
						eucl2eucl(egomo(ti).pnp.src,inv(M(:,:,ti))),...
						'Color', repmat(Ij(sub2ind(size(Ij),round(egomo(ti).pnp.dst(:,2))+1,round(egomo(ti).pnp.dst(:,1))+1)),1,3)...
				);
				
				if ~exist('pc'), pc = pc_i;
				else
					pc = pointCloud([pc.Location;pc_i.Location],'Color',[pc.Color;pc_i.Color]);
				end
				pcshow(pc); drawnow;
			end
		else
			Mij = [];
		end
		
		stats.nepi(ti-t0+1) = size(idx,1);
		stats.npnp(ti-t0+1) = size(Ipnp,1);
		
		fprintf('%d -> %d : #pnp=%d #epi=%d\n',ti,tj,stats.npnp(ti-t0+1),stats.nepi(ti-t0+1));
		
		% solve the ego-motion
		if ti == t0 && hasMotion
			Mij = cam.M(:,:,tj) * inv(cam.M(:,:,ti));
		else
			[Mij,Eepi,Epnp] = solveEgomo(egomo(ti),alpha,Mij);
			if hasMotion
				Mij0 = cam.M(:,:,tj) * inv(cam.M(:,:,ti));
				
				%[Mij,Eepi,Epnp] = solveEgomo(egomo(ti),alpha,Mij0);
				%Mij
				
				dMij = inv(Mij) * Mij0;
				drift = 100 * norm(dMij(1:3,4)) / norm(Mij0(1:3,4));
				%Mij0

				M0j = Mij * M(:,:,ti);
				dM0j = cam.M(:,:,tj) * inv(M0j);
				stats.te(ti-t0+1) = 100 * norm(dM0j(1:3,4)) / norm(cam.M(1:3,4,tj));

				fprintf('drift : %.02f%% / overall: %.02f%%\n',drift,stats.te(ti-t0+1));
				
				if drift > inf
					im = repmat(Ii,1,1,3); im(:,:,3) = Ij; im(:,:,2) = im(:,:,2) * 0.5 + im(:,:,3) * 0.5;
					[Mij0,Eepi0,Epnp0] = solveEgomo(egomo(ti),alpha,Mij0,[],true);
					Iepi = find(Eepi0 > Eepi); Ipnp = find(Epnp0 > Epnp);
					figure, plot(egomo(ti).epi.uid,Eepi0-Eepi,'bx'); grid on; title 'EPI Comparison';
					figure, plot(egomo(ti).pnp.uid,Epnp0-Epnp,'bx'); grid on; title 'RPE Comparison';
					fprintf('ground truth better: pnp=%.02f%% epi=%.02f%%\n',100*numel(find(Eepi0 < Eepi))/numel(Eepi0),100*numel(find(Epnp0 < Epnp))/numel(Epnp));
					
					egomo0.K       = egomo(ti).K;
					egomo0.K_inv   = egomo(ti).K_inv;
					egomo0.epi.src = egomo(ti).epi.src(Iepi,:);
					egomo0.epi.dst = egomo(ti).epi.dst(Iepi,:);
					egomo0.epi.uid = egomo(ti).epi.uid(Iepi,:);
					egomo0.epi.w   = egomo(ti).epi.w  (Iepi,:);
					egomo0.epi.val = Eepi0(Iepi) - Eepi(Iepi);
					
					egomo0.pnp.src = egomo(ti).pnp.src(Ipnp,:);
					egomo0.pnp.dst = egomo(ti).pnp.dst(Ipnp,:);
					egomo0.pnp.uid = egomo(ti).pnp.uid(Ipnp,:);
					egomo0.pnp.w   = egomo(ti).pnp.w  (Ipnp,:);
					egomo0.pnp.val = Epnp0(Ipnp) - Epnp(Ipnp);
					
					solveEgomo(egomo0,alpha,Mij,im,true);
					
					% back-trace
					for tb = ti : ti
						ta = tb - 1;
						Mab = M(:,:,tb) * inv(M(:,:,ta));
						[~,Iepi] = intersect(egomo(ta).epi.uid,egomo0.epi.uid);
						[~,Ipnp] = intersect(egomo(ta).pnp.uid,egomo0.pnp.uid);
						
						egomoa.K       = egomo(ta).K;
						egomoa.K_inv   = egomo(ta).K_inv;
						egomoa.epi.src = egomo(ta).epi.src(Iepi,:);
						egomoa.epi.dst = egomo(ta).epi.dst(Iepi,:);
						egomoa.epi.uid = egomo(ta).epi.uid(Iepi,:);
						egomoa.epi.w   = egomo(ta).epi.w  (Iepi,:);
						
						egomoa.pnp.src = egomo(ta).pnp.src(Ipnp,:);
						egomoa.pnp.dst = egomo(ta).pnp.dst(Ipnp,:);
						egomoa.pnp.uid = egomo(ta).pnp.uid(Ipnp,:);
						egomoa.pnp.w   = egomo(ta).pnp.w  (Ipnp,:);

						Ia = loadImage(cam,ta);
						Ib = loadImage(cam,tb);
						im = repmat(Ii,1,1,3); im(:,:,3) = Ij; im(:,:,2) = im(:,:,2) * 0.5 + im(:,:,3) * 0.5;
						solveEgomo(egomoa,alpha,Mab,im,true);
					end
					
					error 'GAME OVER';
				end
			end
		end
		
		% feature recovery
		%{
		missed = setdiff(1:Fi.NumOfFeatures,idx(:,1));
		missed = missed(find(Fi.G(missed,3) > 0));

		if ~isempty(missed)
			x2dj = homo2eucl(eucl2homo(Fi.G(missed,:)) * (K * Mij(1:3,:))');
			Ij = loadImage(cam,tj);
			inside = find(x2dj(:,1) > 0 & x2dj(:,1) < size(Ij,2) & x2dj(:,2) > 0 & x2dj(:,2) < size(Ij,1));
			x2dj = double(x2dj(inside,:));
			missed = missed(inside);
			desc = extractFeatures(Ij,x2dj+1,'Method','SURF');
			assert(numel(desc) == numel(Fi.V(missed,:)));
			dist = ssd(desc,Fi.V(missed,:));

			%recovered = find(dist < max(cost));
			recovered = find(dist < 0.2);
			I = missed(recovered);

			desc = desc(recovered,:);
			x2di = double(Fi.KP(I,:));
			x2dj = x2dj(recovered,:);

			idx = [idx; I', (1:numel(I))'+numel(Fj.U)];
			
			Fj.V  = [Fj.V;  desc];
			Fj.KP = [Fj.KP; x2dj];
			Fj.U  = [Fj.U;  Fi.U(I)];

			egomo(ti).epi.uid = [egomo(ti).epi.uid; Fi.U(I,:)];
			egomo(ti).epi.src = [egomo(ti).epi.src; x2di];
			egomo(ti).epi.dst = [egomo(ti).epi.dst; x2dj];
			egomo(ti).pnp.uid = [egomo(ti).pnp.uid; Fi.U(I,:)];
			egomo(ti).pnp.src = [egomo(ti).pnp.src; Fi.G(I,:)];
			egomo(ti).pnp.dst = [egomo(ti).pnp.dst; x2dj];
			egomo(ti).pnp.w   = [egomo(ti).pnp.w;   Fi.W(I,:)];
			
			fprintf('%d out of %d features recovered\n',numel(recovered),numel(missed));
		end
		%}

		% update features' 3D coordinates
		Pi = cam.K * [eye(3),zeros(3,1)];
		Pj = cam.K * Mij(1:3,:);

		x3di = Fi.G(idx(:,1),:);
		w3di = Fi.W(idx(:,1),:);
		[x3dj,e3dj,k,rpe] = triangulatePoints(egomo(ti).epi.src,egomo(ti).epi.dst,Pi,Pj);
		%e3dj = sum(rpe.^2,2);
		e3dj = max(rpe,[],2);
		w3dj = 1./(1+e3dj);
		w3dj(find(e3dj > 5*epsilon)) = 0;
		
		%[x3dj,e3dj] = triangulate(egomo.epi.src,egomo.epi.dst,Pi',Pj');
		
		[x3dj,w3dj] = mergePoints(x3di,x3dj,w3di,w3dj);
		Fj.G(idx(:,2),:) = eucl2eucl(x3dj,Mij);
		Fj.W(idx(:,2),:) = w3dj;

		% iterate to the next frame!
		M(:,:,tj) = Mij * M(:,:,ti);
		Ii = Ij;
		Fi = Fj;	
    end
	fprintf('..DONE\n');
end

function y = ssd(x0,x1)
	x0 = bsxfun(@times,x0,1./sqrt(sum(x0.^2,2)));
	x1 = bsxfun(@times,x1,1./sqrt(sum(x1.^2,2)));
	y = sum((x0-x1).^2,2);
end

function hint(seq,cam)
	imageFileRoot = fileparts(cam.ImageFiles{1});
	featurePath   = fullfile(seq.Paths.seqPath,'features','sift');

	fprintf('\n');
	fprintf('Image feature files are missing; please use im2features:\n');
	fprintf('example: im2features \"%s\" \"%s\" -x SIFT\n\n',imageFileRoot,featurePath);
end

function [idx,Fj,matchTime,filterTime,matched,recovered,rejected] = matchGoodFeatures(Fi,Fj,K,Ii,Ij,epsilon,confidence)
	binary = strcmpi(Fi.Metric,'HAMMING');
	tic;
	if binary
		idx = matchFeatures(binaryFeatures(Fi.V),binaryFeatures(Fj.V),'Unique',true,'MatchThreshold',0.5);
	else
		idx = matchFeatures(Fi.V,Fj.V,'Metric',Fi.Metric,'Unique',true,'MatchThreshold',0.5);
	end
	matchTime = toc;
	matched = size(idx,1);

	if ~isempty(Ii) && ~isempty(Ij)
		klt = vision.PointTracker(...
			'MaxBidirectionalError', epsilon, ...
			'BlockSize',             [5,5]);

		missed = setdiff([1:Fi.NumOfFeatures]',idx(:,1));
		xi = double(Fi.KP(missed,:));
		initialize(klt,xi+1,Ii);
		[xj,tracked] = step(klt,Ij);
		tracked = find(tracked);

		xj = xj(tracked,:)-1;
		Fj.KP = vertcat(Fj.KP,xj);
		Fj.U  = vertcat(Fj.U,zeros(numel(tracked),1,'uint32'));
		Fj.V  = vertcat(Fj.V,Fi.V(missed(tracked),:));

		aug = [1:numel(tracked)]'+Fj.NumOfFeatures;
		idx = vertcat(idx,[missed(tracked),aug]);
		recovered = numel(tracked);
		Fj.NumOfFeatures = Fj.NumOfFeatures + recovered;
		
		fprintf('%d features augmented\n',numel(aug));
	end

	% find inliers among the matched features using fundamental matrix
	xi = double(Fi.KP(idx(:,1),:));
	xj = double(Fj.KP(idx(:,2),:));
	%[fmat,inliers] = estimateFundamentalMatrix(xi,xj,'Method','RANSAC','NumTrials',100,'DistanceThreshold',epsilon,'Confidence',99.999);

	tic;
	[~,inliers,err] = estimateEssentialMatrix(xi,xj,K,K,epsilon,confidence,1000);
	filterTime = toc;

	inliers = find(inliers);
	rejected = size(idx,1) - numel(inliers);
	idx = idx(inliers,:);

	if exist('aug')
		all  = [1:Fj.NumOfFeatures]';
		out  = setdiff(aug,idx(:,2));
		keep = setdiff(all,out);
		new  = zeros(Fj.NumOfFeatures,1,'uint32');
		new(keep)  = [1:numel(keep)]';

		Fj.KP = Fj.KP(keep,:);
		Fj.V  = Fj.V(keep,:);
		Fj.U  = Fj.U(keep,:);
		Fj.NumOfFeatures = numel(keep);
		
		idx(:,2) = new(idx(:,2));
		
		assert(numel(find(idx(:,2) == 0)) == 0);
		fprintf('%d augmented features rejected\n',numel(out));
	end
end

function [M,inliers] = estimateMotion(x0,x1,K,epsilon,confidence,iter)
	x0 = eucl2homo(x0);
	[M,err,inliers] = ransac(...
		size(x0,1),	...
		6,			...
		@(idx) solvePnP(x0(idx,:),x1(idx,:)), 					 ...
		@(M)   sqrt(sum((homo2eucl(x0*(K*M(1:3,:))')-x1).^2,2)), ...
		epsilon,	...
		confidence, ...
		iter,       ...
		0.8         ...
	);
	
	function M = solvePnP(x0,x1)
		M = eye(4);
		[M(1:3,1:3),M(1:3,4)] = efficient_pnp(x0,eucl2homo(x1),K);
	end
end

function [U,uid] = allocateNewIds(U,uid)
	idx = find(U == 0);
	U(idx) = uid + (1:numel(idx));
	uid = uid + numel(idx);
end

function [g,w] = mergePoints(gi,gj,wi,wj)
	w = wi + wj;
	g = bsxfun(@times, bsxfun(@times,gi,wi) + bsxfun(@times,gj,wj), 1./w);
end

%{
function [M,g] = solveMotion(x3di,x2di,x2dj,F,cam)
	zi  = x3di(:,3);
	idx = find(isfinite(zi));

	% initialisation using essential matrix decomposition
	E = cam.K' * F * cam.K;
	M = emat2mot(E);

	% pick the best estimation
	numPosK = 0;
	bestIdx = 0;
	for i = 1 : size(M,3)
		[gi,ki] = triangulatePoints(x2di,x2dj,cam,cam,M(:,:,i));
		if i == 1 || numel(find(ki>0)) > numPosK
			bestIdx = i;
			numPosK = numel(find(ki>0));
			g = gi;
		end
	end
	M = M(:,:,bestIdx);

	% No 3D coordinates available at all?
	% Use the result of fundamental matrix decomposition
	%  when no Euclidean data avaiable at all
	if numel(idx) == 0, return; end
	
	% do RPE minimisation using both 3D-to-2D and 2D-to-2D correspondences
	alpha = 0.5;
	optim = optimoptions('lsqnonlin','Display','none');
	x0 = vectorise(M);
    x  = lsqnonlin(@f,x0,[],[],optim);
	M  = devectorise(x);

	function y = f(x)
		y_rpe = f_rpe(x);
		y_epi = f_epi(x);
		y = [(1-alpha)*y_rpe/numel(y_rpe); alpha*y_epi/numel(y_epi)];
	end

	function y = f_rpe(x)
		M = devectorise(x);
		y = x2dj(idx,:) - points2image(eucl2eucl(x3di(idx,:),M),cam);
		y = sqrt(y(:,1).^2 + y(:,2).^2);
	end

	function y = f_epi(x)
		M = devectorise(x);
		F = cameraMotion2fmat(cam,cam,M);
		y = computeSampsonError(x2di,x2dj,F);
	end
end

function M = emat2mot(E)
	% Fix E to be an ideal essential matrix
	[U, D, V] = svd(E);
	e = (D(1,1) + D(2,2)) / 2;
	D(1,1) = e;
	D(2,2) = e;
	D(3,3) = 0;
	E = U * D * V';

	[U, ~, V] = svd(E);

	W = [0 -1 0; 1 0 0; 0 0 1];
	Z = [0 1 0; -1 0 0; 0 0 0];

	% Possible rotation matrices
	R1 = U * W * V';
	R2 = U * W' * V';

	% Force rotations to be proper, i. e. det(R) = 1
	if det(R1) < 0
		R1 = -R1;
	end

	if det(R2) < 0
		R2 = -R2;
	end

	% Translation vector
	Tx = U * Z * U';
	t = [Tx(3, 2), Tx(1, 3), Tx(2, 1)]';

	R = cat(3, R1, R1, R2, R2);
	t = cat(3, t, -t, t, -t);

	M = repmat(eye(4),1,1,4);
	M(1:3,1:4,:) = [R,t];
end

% borrowed from triangulateMidPoint in cameraPose.m
function [g,k] = triangulatePoints(x0,x1,cam0,cam1,M)
	assert(numel(x0) == numel(x1));

	[M0,c0] = pmat2kc(cam0.P);
	[M1,c1] = pmat2kc(cam1.P * M);
	
	x0 = eucl2homo(x0) * inv(M0)';
	x1 = eucl2homo(x1) * inv(M1)';
	t  = c1 - c0;
	%g  = zeros(size(x0,1),3,'like',x1);
	%k  = zeros(size(x0,1),2,'like',x1);
	g   = cell(size(x0,1),1);
	k   = cell(size(x0,1),1);

	for i = 1 : size(g,1)
		A = [x0(i,:)',-x1(i,:)'];
		%k(i,:) = (inv(A'*A) * A' * t)';
		%g0 = c0' + k(i,1) * x0(i,:);
		%g1 = c1' + k(i,2) * x1(i,:);
		ki = (inv(A'*A) * A' * t)';
		g0 = c0' + ki(1) * x0(i,:);
		g1 = c1' + ki(2) * x1(i,:);

		%g(i,:) = (g0 + g1) / 2;
		g{i} = (g0 + g1) / 2;
		k{i} = ki;
	end

	k = cat(1,k{:});
	g = cat(1,g{:});
	
	function [M,c] = pmat2kc(P)
		M = P(1:3,1:3);
		c = -inv(M) * P(1:3,4);
	end
end

function g = mergeValidPoints(g0,g1)
	g  = zeros(size(g0),'like',g0);
	i0 = find(isfinite(g0(:,3)));
	i1 = find(isfinite(g1(:,3)));
	i01 = intersect(i0,i1);
	i0 = setdiff(i0,i01);
	i1 = setdiff(i1,i01);
	g(i0, :) = g0(i0,:);
	g(i1, :) = g1(i1,:);
	g(i01,:) = g0(i01,:) * 0.5 + g1(i01,:) * 0.5;
end

function F = cameraMotion2fmat(cam0,cam1,M)
    K0_inv = inv(cam0.K);
    K1_inv = inv(cam1.K);
    M = cam1.E * M * inv(cam0.E);
    F = K1_inv' * motion2fmat(M) * K0_inv;
end

function F = motion2fmat(M)
    R = M(1:3,1:3);
    t = normalise(M(1:3,4));
    F = skewsymm(t) * R;

    function x = skewsymm(x)
        x = [0,-x(3),x(2);x(3),0,-x(1);-x(2),x(1),0];
    end
end

function y = computeSampsonError(x0,x1,F)
    x0  = eucl2homo(x0);
    x1  = eucl2homo(x1);
    xFx = dot(x1*F,x0,2);
    Fx0 = x0 * F'; nx0 = Fx0(:,1).^2 + Fx0(:,2).^2;
    Fx1 = x1 * F ; nx1 = Fx1(:,1).^2 + Fx1(:,2).^2;
    y   = (xFx .^ 2) ./ (nx0 + nx1);
end

% convert post matrix to a 6-vector
function x = vectorise(M)
    R = M(1:3,1:3);
	[a1,a2,a3] = dcm2angle(R);
    t = M(1:3,4);
	x = [rad2deg([a1;a2;a3]);t];
end

% restore pose matrix from it's vector representation
function M = devectorise(x)
    t = x(4:6); % * 100;
    n = norm(x(1:3));
	a = deg2rad(x(1:3)); 
    R = angle2dcm(a(1),a(2),a(3));
	M = [R,t;0,0,0,1];
end

% vector normalisation
function x = normalise(x)
    x = x / norm(x);
end
%}
