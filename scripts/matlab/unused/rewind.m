function rewind(cam,out)
	t0 = 1;
	tn = numel(cam.ImageFiles);
	frames = tn - t0;
	backrate = 7;
	stopframes = 60;

	out = vision.VideoFileWriter(out,'FrameRate',30);
	fig = vision.VideoPlayer();
	
	for t = t0 : tn
		im = loadImage(cam,t);
		step(out,im);
		step(fig,im);
	end

	for t = 1 : stopframes
		step(out,im);
		step(fig,im);
	end
	
	for t = flipdim(t0:backrate:tn,2)
		alpha = (tn - t) / tn;
		im = loadImage(cam,t);
		im = im * (1-alpha) + repmat(rgb2gray(im),1,1,3) * alpha;
		step(out,im);
		step(fig,im);
	end
end