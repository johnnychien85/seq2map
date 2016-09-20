function [D,G1] = coaxflow(I0,I1,M,K)
	[m,n,~] = size(I1);
	[T0,T1,R1] = estimateCoaxTransforms(K,M(1:3,1:3),M(1:3,4));
	R1i = R1';

	%figure, imshow(imwarp(I0,T0));
	%figure, imshow(imwarp(I1,T1));
	ref2d = imref2d([m,n]);

	W0 = imwarp(I0,T0,'Outputview',ref2d);
	W1 = imwarp(I1,T1,'Outputview',ref2d);

	[P0,A0,R0] = impolar(W0);
	[P1,A1,R1] = impolar(W1);

	explain(W0,W1,P0,P1,max(R0(:)),max(R1(:)));
	
	%figure, imshow([I0,I1],[]);
	%figure, imshow(imfuse(P0,P1));
	w = 2;
	PA0 = [P0(end-w+1:end,:);P0;P0(1:w,:)];
	PA1 = [P1(end-w+1:end,:);P1;P1(1:w,:)];

	DA = disparity(uint8(PA1),uint8(PA0),'BlockSize',2*w+1);
	D = DA(w+1:end-w,:);

	b = norm(M(1:3,4));
	f = K(1,1);
	Z = b * (R1 ./ D - 1);
	R = Z .* R1 / f;
	X = R .* cosd(A1);
	Y = R .* sind(A1);

	bad = find(~isfinite(Z));
	X(bad) = 0;
	Y(bad) = 0;
	Z(bad) = 0;

	[x,y] = meshgrid((1:n)-(n+1)/2,(1:m)-(m+1)/2);
	%figure, imshow(D,[0 16]);
	T1i = invert(T1);
	D = imwarp(imdepolar(D,A1,R1,x,y),T1i,'Outputview',ref2d);
	X = imwarp(imdepolar(X,A1,R1,x,y),T1i,'Outputview',ref2d);
	Y = imwarp(imdepolar(Y,A1,R1,x,y),T1i,'Outputview',ref2d);
	Z = imwarp(imdepolar(Z,A1,R1,x,y),T1i,'Outputview',ref2d);

	G1 = zeros([size(I1),3]);
	for i = 1 : 3
		G1(:,:,i) = R1i(i,1) * X + R1i(i,2) * Y + R1i(i,3) * Z;
	end
	
	%cmap = uint8(255*hot(255));
	%dmin = 0;
	%dmax = 64;
	%cind = round(max(0,min(1,(D-dmin)/(dmax-dmin)))*(size(cmap,1)-1))+1;
	%D = uint8(ind2rgb(cind,cmap));
	%imshow(D*0.7 + repmat(uint8(I1),1,1,3)*0.3);
end

function explain(W0,W1,P0,P1,r0,r1)
	imwrite(circut(W0,r0),'W0.png');
	imwrite(circut(W1,r1),'W1.png');

	P01 = repmat([uint8(P0),uint8(P1)],1,1,3);
	for i = 1 : round(size(P01,1)/10) : size(P01,1)
		P01(i,:,:) = repmat(reshape([0,255,0],1,1,3),1,size(P01,2));
	end
	imwrite(P01,'P01.png');

	function im = circut(im,radius)
		[m,n] = size(im);
		mc = round(round(m+1)/2);
		nc = round(round(n+1)/2);
		im = im(max(1,mc-radius):min(m,mc+radius),max(1,nc-radius):min(n,nc+radius),:);
	end
end

function [T0,T1,R1] = estimateCoaxTransforms(K,R,t)
	t = -R'*t;
	x = t(1);
	y = t(2);
	z = t(3);
	r = norm([x,z]);
	azimuth   = atan2(x,z);
	elevation = atan2(y,r);

	R0 = angle2dcm(0,-azimuth,elevation,'ZYX')';
	R1 = R0 * R';

	K_inv = inv(K);
	H0 = K * R0 * K_inv;
	H1 = K * R1 * K_inv;

	T0 = projective2d(H0');
	T1 = projective2d(H1');
end

function [im,t,r] = impolar(im,theta,radii)
	if nargin < 2, theta = 0:359; end
	if nargin < 3, radii = getOptimalRadii(im); end

	[m,n] = size(im);
	[r,t] = meshgrid(radii,theta);
	[x,y] = pol2cart(deg2rad(t),r);
	x = x + (n + 1) / 2;
	y = y + (m + 1) / 2;

	%figure, imshow(im), hold on;
	%plot(x,y,'r.');
	
	im = interp2(double(im),x,y,'cubic',0);
end

function im = imdepolar(im,theta,radii,x,y)
	[t,r] = cart2pol(x,y);
	im = [im;im];
	theta = [theta-360;theta];
	radii = [radii;radii];
	im = interp2(radii,theta,double(im),r,rad2deg(t),'cubic',0);
end

function radii = getOptimalRadii(im)
	[m,n] = size(im);
	rmin = 0;
	rmax = min(m,n)/2;
	radii = rmin:rmax;
end