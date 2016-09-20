function featureShow(cam,out)
	fps = 30;
	scantime = 1;
	locktime = 1;
	showtime = 2;
	radius0 = 32;
	radius1 = 2;

	im = loadImage(cam,1,'greyscale');

	out = vision.VideoFileWriter(out,'FrameRate',fps);
	fig = vision.VideoPlayer();

	scanframes = scantime * fps;
	y0 = 1;
	h = round(size(im,1) / scanframes);
	mask = zeros(size(im),'like',im);
	mask = repmat(mask,1,1,3);

	for t = 1 : scanframes
		y1 = min(y0+h,size(im,1)-1);
		mask = mask * 0.8;
		mask(y0:y1,:,2) = 255;
		canvas = max(repmat(im,1,1,3),mask);
		y0 = y1 + 1;
		step(fig,canvas);
		step(out,canvas);
	end
	
	while max(mask(:)) > 0
		canvas = max(repmat(im,1,1,3),mask);
		mask = mask * 0.3;
		step(fig,canvas);
		step(out,canvas);
	end

	showframes = showtime * fps;
	fi = cam.Features(1).F;
	fj = cam.Features(2).F;
	idx = matchFeatures(fi,fj);
	ii = idx(:,1);
	ij = idx(:,2);
	i = cam.Features(1).G(idx(:,1),:);
	xi = double(cam.Features(1).KP(ii,:) - 1);
	xj = double(cam.Features(2).KP(ij,:) - 1);
	[fmat,matched] = estimateFundamentalMatrix(xi,xj);
	matched = find(matched);
	ii = ii(matched,:);
	ij = ij(matched,:);
	xi = xi(matched,:);
	xj = xj(matched,:);

	x = xi;
	r = inf(size(x,1),1);
	h = round(size(im,1) / showframes);
	dr = (radius1 - radius0) / (locktime * fps);
	y0 = 1;

	picks = ceil(numel(r) / showframes);
	
	for t = 1 : showframes
		r = max(r + dr, radius1);
		y1 = min(y0+h,size(im,1)-1);

		%idx = find(x(:,2) >= y0 & x(:,2) < y1);
		idx = pick(find(~isfinite(r)),picks);

		r(idx) = radius0;
		y0 = y1 + 1;
		
		idx = find(isfinite(r));
		canvas = repmat(im,1,1,3);
		canvas = canvas * 0.5 + insertShape(canvas,'circle',[x(idx,1:2),r(idx)],'color','green','linewidth',2,'opacity',rand(1)) * 0.5;
		
		step(fig,canvas);
		step(out,canvas);
	end
	
	r(find(~isfinite(r))) = radius0;

	while max(r) > radius1
		r = max(r + dr, radius1);
		canvas = repmat(im,1,1,3);
		canvas = canvas * 0.5 + insertShape(canvas,'circle',[x(idx,1:2),r(idx)],'color','green','linewidth',2,'opacity',0.1) * 0.5;
		step(fig,canvas);
		step(out,canvas);
	end
end

function idx = pick(idx,n)
	I = randperm(numel(idx));
	I = I(1:min(n,numel(I)));
	idx = idx(I);
end