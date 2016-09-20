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