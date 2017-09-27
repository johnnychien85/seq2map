function map = loadMap(from)
    lmkPath = fullfile(from,'landmarks.dat');
    hitPath = fullfile(from,'hits.dat');

    % load landmarks
    d = dir(lmkPath);
    n = 9;
    m = d.bytes / n / 8;
    f = fopen(lmkPath,'r');
    map.lmk = fread(f,[n,m],'double')';
    fclose(f);

    % load hits
    d = dir(hitPath);
    n = 3 + 1 + 2;
    m = d.bytes / (4*8 + 2*8);

    map.hits = struct(...
        'lmk', zeros(0,1,'uint64'),...
        'frm', zeros(0,1,'uint64'),...
        'src', zeros(0,1,'uint64'),...
        'idx', zeros(0,1,'uint64'),...
        'prj', zeros(0,2,'double') ...
    );

    map.hits(m).prj = [0,0];
    
    f = fopen(hitPath,'r');
    for i = 1 : m
		map.hits(i).lmk = fread(f,1,'ubit64');
        map.hits(i).frm = fread(f,1,'ubit64');
        map.hits(i).src = fread(f,1,'ubit64');
        map.hits(i).idx = fread(f,1,'ubit64');
        map.hits(i).prj = fread(f,[1,2],'double');
	end
    fclose(f);
end