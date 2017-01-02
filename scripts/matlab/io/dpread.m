function dp = dpread(path,dmax)
	dp = single(imread(path)) / 65535 * dmax;
end