function rio(seq)
	cam = seq.cam(1);

	if ~isfield(cam,'Features')
		error 'Image features are not available; use findSequenceFeatures first.';
	end
	
	t0 = 1;
	tn = numel(cam.ImageFiles);
	frames = tn - t0;
	
	% initialise motion matrix stack
    M = zeros(4,4,tn);
	M(:,:,1) = eye(4);

	% initialise features' 3D coordinates of the first frame
	cam.Features(t0).G = inf(cam.Features(t0).NumOfFeatures,3);

	fig = vision.VideoPlayer();
	whitebg(figure('Position',[0,0,320,240])); a = gca; axis equal; grid on; hold on;
	xlabel 'x', ylabel 'y', zlabel 'z'; view([0,-1,0]);
	cmap = jet(256);
	
	for t = t0 : tn
		ti = t;
		tj = t + 1;
		
		% features of ti and tj
		fi = cam.Features(ti).F;
		fj = cam.Features(tj).F;

		% find matched features
		idx = matchFeatures(fi,fj);
		ii  = idx(:,1);
		ij  = idx(:,2);
		gi  = cam.Features(ti).G(idx(:,1),:);
		xi  = double(cam.Features(ti).KP(ii,:) - 1);
		xj  = double(cam.Features(tj).KP(ij,:) - 1);

		% find inliers among the matched features using fundamental matrix
		[fmat,matched] = estimateFundamentalMatrix(xi,xj,'DistanceThreshold',1e-4);
		matched = find(matched);
		ii = ii(matched,:);
		ij = ij(matched,:);
		gi = gi(matched,:);
		xi = xi(matched,:);
		xj = xj(matched,:);

		% solve the ego-motion
		[Mij,gj] = solveMotion(gi,xi,xj,fmat,cam);
		M(:,:,tj) = Mij * M(:,:,ti);

		% update features' 3D coordinates
		cam.Features(tj).G = inf(cam.Features(tj).NumOfFeatures,3);
		cam.Features(tj).G(ij,:) = eucl2eucl(mergeValidPoints(gi,gj),Mij);

		%ij = find(gj(:,3) > 0 & gj(:,3) < 100);
		%gj = eucl2eucl(gj(ij,:),inv(M(:,:,ti)));
		%ci = -M(1:3,1:3,ti)' * M(1:3,4,ti);
		%cj = -M(1:3,1:3,tj)' * M(1:3,4,tj);
		%v = gj(:,2);
		%vmin = -15;
		%vmax = +15;
		%cidx = round(max(0,min(1,(v - vmin) ./ (vmax - vmin))) * (size(cmap,1) - 1)) + 1;
		%scatter3(gj(:,1), gj(:,2), gj(:,3), 5, cmap(cidx,:),'filled');
		%plot3([ci(1),cj(1)],[ci(2),cj(2)]-30,[ci(3),cj(3)],'w-','linewidth',5);
		%drawnow;
		
		% dense reconstruction
		%im = imread2(cam.ImageFiles{ti},cam.ImageFiles{tj});
		im = imread2(cam.ImageFiles{ti},cam.ImageFiles{tj});
		
		if t == t0
			f = figure, imshow(im(:,:,1));
			ply = impoly(gca);
            vtx = ply.getPosition();
            close(f);
			
			tracker = vision.PointTracker;
			initialize(tracker,vtx,im(:,:,1));
		end
		
		vtx = step(tracker,im(:,:,3));

		im = im(:,:,3);
		im = insertShape(im, 'FilledPolygon', reshape(vtx',1,[]), 'Color', 'green', 'opacity', 0.7);
		
		%[Ti,Tj] = estimateMonocularRectification(cam.K,Mij(1:3,1:3),Mij(1:3,4));
		[Ti,Tj] = estimateMonocularRectification(cam.K,M(1:3,1:3,tj),M(1:3,4,tj));
		[m,n,~] = size(im);
		%k = (n - m) / 2;
		%rf = imref2d([n,n],[1,n],[1-k,m+k]);
		rf = imref2d([m,n]);

		[Ji,Bi] = imwarp(im(:,:,1),Ti,'OutputView',rf);
		[Jj,Bj] = imwarp(im(:,:,3),Tj,'OutputView',rf);
%figure, imshow(imfuse(Ji,Jj));
%drawnow; pause;
%		Dij = rdisparity(Ji,Jj);
		
		%Dij = imwarp(Dij,);
		%Dij = disparity(Ji,Jj,'Method','SemiGlobal','BlockSize',5,'DisparityRange',[-64,64],'UniquenessThreshold',0,'ContrastThreshold',0.01);
		

		%[jy,jx] = find(Dij);
		%[ji,jj] = Bi.worldToSubscript(jx,jy);
		
		%idx = find(ji > 0 & ji <= m & jj > 0 & jj <= n);
		%jy  = jy(idx);
		%jx  = jx(idx);
		%Ii  = sub2ind([m,n],jy,jx);
		%Ij  = sub2ind([m,n],ji,jj);
		
		%Kij = zeros(m,n,'uint8');
		%Kij(Ij) = Dij(Ii)+65;
		
		%im = repmat(im(:,:,1),[1,1,3]);
		%im = repmat(Ji,1,1,3);
		%im(:,:,2) = Ji * .5 + Jj * .5;
		%im(:,:,3) = Jj;
		%im = im * 0.5 + uint8(255*ind2rgb(uint8(Dij+65),jet(255))) * 0.5;
		%im = im * 0.5 + uint8(255*ind2rgb(Kij,jet(255))) * 0.5;
		
		%im = Iij*0.8 + repmat(Ii*0.2,1,1,3);
		%%im = insertShape(im,'circle',[x(idx),y(idx),repmat(1,numel(idx),1)]);
		%im = insertShape(im,'line',[xi(matched,:),xj(matched,:)]);

		%Iij = cat(3,im(:,:,1),im(:,:,3));
		%[Iy,Ix,It] = gradient(double(Iij));
		
		%Iij(:,:,1) = uint8(sum(abs(Iy),3));
		%Iij(:,:,2) = uint8(sum(abs(Ix),3));
		%Iij(:,:,3) = uint8(sum(abs(It),3));

		%figure, imshow(Dij,[0 64]);
		
		%Ji = imresize(Ji,[size(im,1),size(im,2)]);
		%Jj = imresize(Jj,[size(im,1),size(im,2)]);

		%im = repmat(Ji,1,1,3);
		%im(:,:,2) = Ji * .5 + Jj * .5;
		%im(:,:,3) = Jj;
			
		step(fig,im);
	end
end

function [T1,T2] = estimateMonocularRectification(K,R,t)
	t = -R'*t / norm(t);
	u = [t(1),   0 , t(3)]';
	v = [  0 , t(2),   0 ]';

	ele = atan2(norm(v),norm(u));
	azi = acos(u(3)/norm(u));
	%t
	%rad2deg([ele,azi])
	R1 = angle2dcm(0,azi,ele,'ZYX')';
	%R1
	K_inv = inv(K);
	H1 = K * R1 * K_inv;
	H2 = K * R1 * R' * K_inv;

	T1 = projective2d(H1');
	T2 = projective2d(H2');
end

function dmap = rdisparity(I1,I2)
	d = 0:64;
	w = 5;

	global cost dmap smap icst;
	cost = zeros([size(I1),numel(d)]);
	dmap = zeros(size(I1));
	for k = 1 : numel(d)
		dk  = d(k);
		I2k = imresize(I2,size(I2)+dk*2);
		I2k = I2k(dk+1:end-dk,dk+1:end-dk);
		cost(:,:,k) = sqrt(conv2(double(I1 - I2k).^2,ones(2*w+1),'same'));
	end

	smap = sum(cost,3);
	icst = (1 - cost ./ repmat(smap,1,1,numel(d))) / (numel(d)-1);
	best = icst(:,:,1);
	for k = 2 : numel(d)
		%dmap = dmap + icst(:,:,k) * d(k);
		idx = find(icst(:,:,d(k)) > best);
		dmap(idx) = d(k);
	end
	figure, imshow(dmap,[]);
	pause;
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

	% No 3D  coordinates available at all?
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

function [im,im0,im1] = imread2(leftFile,rightFile)
	im0 = histeq(imread2grey(leftFile ));
	im1 = histeq(imread2grey(rightFile));
	im  = zeros([size(im0),3], 'uint8');
	im(:,:,1) = im0;
	im(:,:,3) = im1;
	im(:,:,2) = im0 * .5 + im1 * .5;
end

function im = imread2grey(filename)
	im = imread(filename);
	if size(im,3) == 3, im = rgb2gray(im); end
end
