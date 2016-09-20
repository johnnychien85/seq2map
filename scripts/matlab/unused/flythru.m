function flythru(pc,out)
	lookat = mean(pc.Location) + [0,-0.5,-1.5];
	campos = [0,-0.2,-5];
	camup  = [0,-1,0];
	close  = 0.3;
	vpeak  = 0.028;
	vstop  = 0.018;
	
	out = vision.VideoFileWriter(out,'FrameRate',30);

	f = figure('Position',[0,0,1280,720]);
	pcshow(pc,'markersize',13); a = gca;
	whitebg(f);
	
	a.Projection = 'perspective';
	a.CameraViewAngleMode = 'manual';
	a.CameraPositionMode  = 'manual';
	a.CameraTargetMode    = 'manual';
	a.CameraUpVectorMode  = 'manual';
	a.CameraViewAngle = 30;
	a.CameraPosition  = campos;
	a.CameraTarget    = lookat;
	a.CameraUpVector  = camup;

	dpos  = lookat - campos;
	rinit = norm(dpos);
	rstop = close * rinit;

	while norm(campos - lookat) > rstop
		r = norm(campos - lookat);
		v = vstop + (vpeak-vstop) * (cos(pi * (r - rinit) / (rinit - rstop)) + 1) / 2;
		campos = campos + v * dpos / rinit;
		
		a.CameraPosition = campos;
		step(out,frame2im(getframe));
	end
	
	dinit = campos - lookat;
	xinit = campos;
	xstop = lookat + [dinit(1),dinit(2),-dinit(3)];

	while norm(campos - xstop) > 0.1
		v = xstop - campos;
		v = v/norm(v) * 0.006;
		%x = campos + v;
		[vx,vy,~] = cart2sph(v(1),v(2),v(3));
		[v(1),v(2),v(3)] = sph2cart(vx,vy,close);
		campos = campos + v * 0.03;
		%campos = campos + v/norm(v) * 0.006;
		
		a.CameraPosition = campos;
		step(out,frame2im(getframe));
	end
	
	while norm(campos - lookat) < 8
		campos = campos + v/norm(v) * 0.02;

		a.CameraPosition = campos;
		step(out,frame2im(getframe));
	end
	
end