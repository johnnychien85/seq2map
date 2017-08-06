function demo_emat2mot(K)
	n = 50;
	x = rand(n,3)*3;
	x(:,1:2) = bsxfun(@minus,x(:,1:2),mean(x(:,1:2)));
	
	M = eye(4);
	M(1:3,1:3) = angle2dcm(deg2rad(rand(1)*5),deg2rad(rand(1)*10),0)';
	M(1:3,4) = M(1:3,1:3)' * [1,0,0]';

	x0 = homo2eucl(x*K');
	x1 = homo2eucl(eucl2homo(x)*M(1:3,:)'*K');
M
	M = pts2mot(x0,x1,K)
end

function M = pts2mot(x0,x1,K)
	F = estimateFundamentalMatrix(x0,x1,'Method','Norm8Point');
	E = K' * F * K;
	M = chieralityCheck(emat2mot(E),x0,x1,K);
end

function M = emat2mot(E)
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
end

function M = chieralityCheck(M, x0, x1, K)
	numPosK = 0;
	bestIdx = 0;
	
	for i = 1 : size(M,3)
		subplot(2,2,i);
		axis equal; grid on; hold on;
		
		P0 = K * [eye(3),zeros(3,1)];
		P1 = K * M(1:3,:,i);
		[xi,~,ki] = triangulatePoints(x0,x1,P0,P1);
		
		c0 = [0,0,0];
		z0 = [0,0,0.5];
		c1 = [0,0,1,1] * inv(M(:,:,i))';
		z1 = [0,0,0.5,1] * inv(M(:,:,i))';

		idx = find(abs(xi(:,1)) < 1 & abs(xi(:,2)) < 1 & abs(xi(:,3)) < 1);
		
		plot3([c0(1),c1(1)],[c0(2),c1(2)],[c0(3),c1(3)],'r-','linewidth',3);
		quiver3(c0(1),c0(2),c0(3),z0(1)-c0(1),z0(2)-c0(2),z0(3)-c0(3),1,'b-','linewidth',2,'MaxHeadSize',5);
		quiver3(c1(1),c1(2),c1(3),z1(1)-c1(1),z1(2)-c1(2),z1(3)-c1(3),1,'b-','linewidth',2,'MaxHeadSize',5);
		plot3(xi(idx,1),xi(idx,2),xi(idx,3),'k.');
		xlabel 'x'; ylabel 'y'; zlabel 'z';
		
		n = numel(find(ki>0));
		if i == 1 || n > numPosK
			bestIdx = i;
			numPosK = n;
		end
	end
	M = M(:,:,bestIdx);
end