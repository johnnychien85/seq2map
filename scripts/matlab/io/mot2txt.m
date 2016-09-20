function mot2txt(M, filename)
	f = fopen(filename, 'w');
	for t = 1 : size(M,3)
		fprintf(f, '%f %f %f %f %f %f %f %f %f %f %f %f\n', M(1:3,:,t)');
	end
end