function pc = coaxflowTest(cam,M)
	if nargin < 2, M = cam.M; end

	%f = figure;
	cmap = jet(255);
	dlim = [0,1000];

	for t = 1 : size(M,3) - 1
		I0 = loadImage(cam,t,'greyscale');
		I1 = loadImage(cam,t+1,'greyscale');
		Mt = M(:,:,t+1) * inv(M(:,:,t));
		[~,G1] = coaxflow(I0,I1,Mt,cam.K);
		
		%figure(f);
		%dpshow(G(:,:,3),dlim,jet(255),I1);
		%drawnow;

		X = G1(:,:,1);
		Y = G1(:,:,2);
		Z = G1(:,:,3);
		idx = find(Z > dlim(1) & Z < dlim(2));

		G1 = eucl2eucl([X(idx),Y(idx),Z(idx)],inv(M(:,:,t+1)));
		pc1 = pointCloud(G1,'Color',repmat(I1(idx),1,3));

		if ~exist('pc'), pc = pc1;
		else             pc = pcmerge(pc,pc1,10);
		end

		pcshow(pc); drawnow;
	end
end