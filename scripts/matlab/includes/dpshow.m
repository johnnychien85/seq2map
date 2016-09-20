function im = dpshow(dpmap,dlim,cmap,im)
	if nargin < 3, cmap = flipdim(jet(64),1); end
	if nargin >= 4
		if size(im,3) == 1, im = repmat(im,1,1,3); end
	end

	dmin = dlim(1);
	dmax = dlim(2);
	cind = round(max(0,min(1,(dpmap-dmin)/(dmax-dmin)))*(size(cmap,1)-1))+1;
	
	if nargin < 4, im = uint8(ind2rgb(cind,cmap)*255);
	else		   im = uint8(ind2rgb(cind,cmap)*255) * 0.6 + im * 0.4;
	end

	imshow(im); colormap(cmap); colorbar('South'); caxis(dlim);
end