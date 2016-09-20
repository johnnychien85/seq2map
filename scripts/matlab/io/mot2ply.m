function mot2ply(mot,filename,color)
	TA = [0 1 2; 0 2 3; 0 3 4; 0 4 5; 0 5 1];
	TB = [7 1 2; 2 7 8; 8 2 3; 3 8 9; 9 3 4; 4 9 10; 10 4 5; 5 10 11; 11 5 1; 1 11 7];

	TA = flipdim(TA,2);
	TB = flipdim(TB,2);
	
	tn = size(mot,3);
	n = size(TA,1);
	r = 30;
	nVertices = tn * (n + 1);
	nFaces = n * tn + n * (tn - 1);

	M.TRI = cell(nFaces,3);
	M.G = cell(nVertices, 3);
	M.C = cell(nVertices, 3);

	T = deg2rad(linspace(0, 360, n + 1));
	[X,Y] = pol2cart(T(1:n)', r);
	C = [0 0 0; X,Y,zeros(n,1)];
	GI = 1 : (n+1);
	TI = 1 : n;

	tt = 0;
	for t = 1 : tn
		M.G{t} = eucl2eucl(C,invmot(mot(:,:,t)));
		M.C{t} = repmat(color, size(M.G{t},1), 1);
		M.TRI{t} = TA;
		if t < tn
			M.TRI{t} = [M.TRI{t}; TB];
		end
		M.TRI{t} = M.TRI{t} + tt;
		tt = tt + size(M.G{t},1);
	end

	M.G   = cat(1, M.G{:});
	M.TRI = cat(1, M.TRI{:}) + 1;
	M.C   = cat(1, M.C{:});
	mesh2ply(M, filename);
	
	% % output as PLY file
	% fid = fopen(filename, 'w');

	% fprintf(fid, 'ply\n'); % begin of header
	% fprintf(fid, 'format ascii 1.0\n');
	% fprintf(fid, 'element vertex %d\n', tn);
	% fprintf(fid, 'property float x\n');
	% fprintf(fid, 'property float y\n');
	% fprintf(fid, 'property float z\n');
	% fprintf(fid, 'property uchar red\n');
	% fprintf(fid, 'property uchar green\n');
	% fprintf(fid, 'property uchar blue\n');

	% fprintf(fid, 'element edge %d\n', tn-1);
	% fprintf(fid, 'property int vertex1\n');
	% fprintf(fid, 'property int vertex2\n');
	% fprintf(fid, 'property uchar red\n');
	% fprintf(fid, 'property uchar green\n');
	% fprintf(fid, 'property uchar blue\n');
	% fprintf(fid, 'end_header\n'); % end of header

	% for t = 1 : tn
		% g = mot{t,1} * [0 0 0 1]';
		% % fwrite(fid, g(1:3), 'float32');
		% % fwrite(fid, round(color), 'uchar');
		% fprintf(fid, '%.02f %.02f %.02f %u %u %u\n', g(1:3), round(color));
	% end

	% for t = 1 : tn - 1
		% % fwrite(fid, [t-1 t], 'int');
		% % fwrite(fid, round(color), 'uchar');
		% fprintf(fid, '%d %d %u %u %u\n', [t-1 t], round(color));
	% end	

	% fclose(fid);
end
