function mesh2ply(M, filename)
	if ~isfield(M, 'TRI'), M.TRI = []; end;
	if ~isfield(M, 'C') || isempty(M.C),
		M.C = zeros(size(M.G,1),3);
		C = gray(255);
		Zmin = min(M.G(:,3));
		Zmax = max(M.G(:,3));
		CIdx = round((M.G(:,3) - Zmin) / (Zmax - Zmin) * (size(C,1) - 1)) + 1;
		M.C = uint8(255 * C(CIdx,:));
	end;

	% output as PLY file
	fid = fopen(filename, 'w');

	fprintf(fid, 'ply\n'); % begin of header
	fprintf(fid, 'format binary_little_endian 1.0\n');
	% fprintf(fid, 'format ascii 1.0\n');
	fprintf(fid, 'element vertex %d\n', size(M.G,1));
	fprintf(fid, 'property float x\n');
	fprintf(fid, 'property float y\n');
	fprintf(fid, 'property float z\n');
	fprintf(fid, 'property uchar red\n');
	fprintf(fid, 'property uchar green\n');
	fprintf(fid, 'property uchar blue\n');

	fprintf(fid, 'element face %d\n', size(M.TRI,1));
	fprintf(fid, 'property list uchar int vertex_indices\n');
	fprintf(fid, 'end_header\n'); % end of header

	fprintf('Writing coordinates...');
	for i = 1 : size(M.G,1)
		% fprintf(fid, '%f %f %f %u %u %u\n', M.G(i,:), round(M.C(i,:)));
		fwrite(fid, M.G(i,:), 'float32');
		fwrite(fid, round(M.C(i,:)), 'uchar');
		if mod(i,floor(size(M.G,1)/10)) == 0, fprintf('..%d%%', floor(100*i/size(M.G,1))); end
	end
	fprintf('..DONE\n');

	fprintf('Writing triangles...');	
	for i = 1 : size(M.TRI,1)
		% fprintf(fid, '3	%d %d %d\n', M.TRI(i,1)-1, M.TRI(i,2)-1, M.TRI(i,3)-1);
		fwrite(fid, 3, 'uchar');
		fwrite(fid, [M.TRI(i,1)-1, M.TRI(i,2)-1, M.TRI(i,3)-1], 'int');
		if mod(i,floor(size(M.TRI,1)/10)) == 0, fprintf('..%d%%', floor(100*i/size(M.TRI,1))); end
	end	
	fprintf('..DONE\n');

	fprintf('Mesh dumped consists of %d vertices and %d triangles\n', size(M.G,1), size(M.TRI,1));
	fclose(fid);
end