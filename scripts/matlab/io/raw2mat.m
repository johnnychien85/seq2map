function x = raw2mat(path, m, n, s)
	if nargin < 3, s = 'int16'; end;
	f = fopen(path, 'r');
	x = fread(f, [n m], s)';
	fclose(f);
end