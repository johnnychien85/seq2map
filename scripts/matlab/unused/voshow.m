function voshow(cam,out)
	t0 = 1;
	tn = numel(cam.ImageFiles);
	fps = 30;
	trace = 0;
	tmmap = 30;

	out = vision.VideoFileWriter(out,'FrameRate',fps);
	fig = vision.VideoPlayer();

	if tmmap < inf
		fgm = figure('Position',[0,0,320,360]);
		whitebg(fgm); a = gca; axis equal; grid on; hold on;
		xlabel 'x', ylabel 'y', zlabel 'z'; view([0,-1,0]); zlim([0,250]);
	end
	
	if trace > 0
		cmap = hot(trace);
	end
	
	for t = t0 : tn - 1
		im = loadImage(cam,t,'greyscale');
		canvas = repmat(im,1,1,3);
		
		if trace > 0
			for ti = max(t0,t-trace+1) : t
				xi = cam.Features(ti).KP;
				idx = find(isfinite(cam.Features(ti).G(:,3)));
				%idx = find(cam.Features(ti).G(:,3) < 3);
				xi  = xi(idx,:);
				r   = 1+repmat(trace+ti-t,numel(idx),1)*0.5;
				o   = 1 / (1+trace+ti-t);
				%canvas = insertShape(canvas,'FilledCircle',[xi,r],'color',uint8(255*cmap(ti-t+trace,:)),'opacity',0.1);
				canvas = insertShape(canvas,'FilledCircle',[xi,r],'color','green','opacity',o);
			end
		end
		
		ti = t;
		tj = t + 1;
		
		ci = -cam.M(1:3,1:3,ti)' * cam.M(1:3,4,ti);
		cj = -cam.M(1:3,1:3,tj)' * cam.M(1:3,4,tj);
		
		if tmmap < inf
			plot3(a,[ci(1),cj(1)],[ci(2),cj(2)]-30,[ci(3),cj(3)],'w-','linewidth',3);
		end
		
		if ti > tmmap
			alpha = min((ti-tmmap)/40,0.8);
			minimap = frame2im(getframe(fgm));
			[m,n,~] = size(minimap);
			y0 = 32;
			x0 = (size(canvas,2) - n) / 2;
			canvas(y0:y0+m-1,x0:x0+n-1,:) = canvas(y0:y0+m-1,x0:x0+n-1,:) * (1-alpha) + minimap * alpha;
		end
		
		step(out,canvas);
		step(fig,canvas);
	end

end