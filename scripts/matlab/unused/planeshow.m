function planeshow(cam,out)
	fps = 22;
	out = vision.VideoFileWriter(out,'FrameRate',fps);
	fig = vision.VideoPlayer();
	frametime = 0.1;
	flashtime = 1;
	warpingtime = 2;

	im = loadImage(cam,1,'greyscale');
	f = figure, imshow(im);
	ply = impoly(gca);
    vtx = ply.getPosition();
	B = uint8(createMask(ply));
    close(f);

	canvas = repmat(im,1,1,3);
	for t = 1 : 4
		i = t;
		j = mod(t,4)+1;
		canvas = insertShape(canvas,'line',[vtx(i,:),vtx(j,:)],'color','green','linewidth',5);
		
		for i = 1 : frametime * fps
			step(fig,canvas);
			step(out,canvas);
		end
	end

	flashframes = flashtime * fps;
	for t = flashframes : -1 : 0
		o = max((t/flashframes)^2,0.1);
		canvas = insertShape(im,'FilledPolygon',reshape(vtx',1,[]),'color','white','opacity',o);
		canvas = insertShape(canvas,'Polygon',reshape(vtx',1,[]),'color','white','linewidth',5);
		step(fig,canvas);
		step(out,canvas);
	end

	% warping animation
	warpingframes = warpingtime * fps;
	[m,n,k] = size(im);
	e = [n*0.3,m*0.3;n*0.7,m*0.3;n*0.7,m*0.7;n*0.3,m*0.7];
	%e = [1,1;n*0.5,1;n*0.5,m*0.5;1,m*0.5];
	u = e - vtx;
	klt = vision.PointTracker();
	initialize(klt,vtx,im);
	im = im .* B;
	for t = 1 : warpingframes
		k = t / warpingframes;
		vtx_t = vtx + k*u;
		tform = fitgeotrans(vtx,vtx_t,'projective');
		ref2d = imref2d([m,n]);
		canvas = imwarp(im,tform,'Outputview',ref2d);
		canvas = insertShape(canvas,'Polygon',reshape(vtx_t',1,[]),'color','white','linewidth',5);
		step(fig,canvas);
		step(out,canvas);
	end

	for t = 2 : 120
		im = loadImage(cam,t,'greyscale');	
		vtx = step(klt,im);
		tform = fitgeotrans(vtx,vtx_t,'projective');
		canvas = imwarp(im,tform,'Outputview',ref2d);
		canvas = insertShape(canvas,'Polygon',reshape(vtx_t',1,[]),'color','white','linewidth',5);
		step(fig,canvas);
		step(out,canvas);
	end
end