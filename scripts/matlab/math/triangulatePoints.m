% borrowed from triangulateMidPoint in cameraPose.m
function [g,e,k,rpe] = triangulatePoints(x0,x1,P0,P1)
	assert(numel(x0) == numel(x1));

	[KR0,c0] = pmat2krc(P0);
	[KR1,c1] = pmat2krc(P1);

	% normalised image coordinates
	n0 = eucl2homo(x0) * inv(KR0)';
	n1 = eucl2homo(x1) * inv(KR1)';

	t = c1 - c0;
	m = c0 + c1;

	g = zeros(3,size(n0,1));
	k = zeros(2,size(n0,1));
	e = zeros(1,size(n0,1));

	for i = 1 : size(g,2)
		A = [n0(i,:)',-n1(i,:)'];
		B = [n0(i,:)', n1(i,:)'];
		k(:,i) = inv(A'*A)*A'*t;
		g(:,i) = (B*k(:,i) + m) / 2;
		e(:,i) = norm(A*k(:,i) - t);
	end

	g = g';
	e = e';
	k = k';

	if nargout > 3
		y0 = homo2eucl(eucl2homo(g) * P0');
		y1 = homo2eucl(eucl2homo(g) * P1');
		rpe = [blkproc(x0-y0,[1,3],'norm'),blkproc(x1-y1,[1,3],'norm')];
	end
end

function [KR,c] = pmat2krc(P)
    KR = P(1:3,1:3);
    c  = -inv(KR) * P(1:3,4);
end