function pts2ply(pts,ply,varargin)
    po = parseInput(pts,ply,varargin{:});

    f = fopen(ply,'wb');
    assert(f ~= -1, 'error opening file for writing');

    if isemptsy(po.colormap),     po.colormap = uint8(jet(64) * 255);
    elseif isfloat(po.colormap), po.colormap = uint8(po.colormap * 255);
    end

    if isemptsy(po.color)
        po.color = computeColorFromZ(pts, size(po.colormap, 1));
        useColormap = true;
    else
        if size(po.color,2) == 1
            a = size(po.colormap,1) - 1;
            b = 1;
            useColormap = true;
        else
            a = 255; % for 24-bit RGB colors
            b = 0;
            useColormap = false;            
        end

        if isfloat(po.color), po.color = uint32(a * po.color + b); end
        assert(all(all(po.color >= b & po.color <= a + b)));
    end

    if useColormap, c = po.colormap(po.color,:);
    else            c = po.color; end

    % write header
    fprintf(f, 'ply\r\n');
    if po.binary, fprintf(f, 'format binary_little_endian 1.0\r\n');
    else          fprintf(f, 'format ascii 1.0\r\n'); end

    fprintf(f, 'element vertex %d\r\n', size(pts,1));
    fprintf(f, 'property float x\r\n');
    fprintf(f, 'property float y\r\n');
    fprintf(f, 'property float z\r\n');
    fprintf(f, 'property uchar red\r\n');
    fprintf(f, 'property uchar green\r\n');
    fprintf(f, 'property uchar blue\r\n');
    fprintf(f, 'end_header\r\n', size(pts,1));

    % write data
    fprintf('Dumping vertices..');
    if po.binary
        for i = 1 : size(pts,1)
            fwrite(f, pts(i,:), 'float');
            fwrite(f, c (i,:), 'uint8');
            pprint(i, size(pts,1));            
        end
    else
        for i = 1 : size(pts,1)
            fprintf(f, '%.02f %.02f %.02f %d %d %d\r\n', pts(i,:), c(i,:));
            pprint(i, size(pts,1));
        end
    end
    fprintf('..DONE\n');
    fclose(f);
end

function po = parseInput(pts, ply, varargin)
    points = size(pts,1);

    po = inputParser;
    addRequired (po, 'points',         @checkPoints);
    addRequired (po, 'filename',       @isstr);
    addParameter(po, 'binary',   true, @islogical);
    addParameter(po, 'color',    [],   @checkColorVector);
    addParameter(po, 'colormap', [],   @checkColormap);
    parse(po, pts, ply, varargin{:});

    po = po.Results;

    function okay = checkPoints(pts)
        okay = (ndims(pts) == 2 && size(pts,2) == 3);
    end

    function okay = checkColorVector(v)
        [m,n] = size(v);
        okay = (isemptsy(v)) || ...
               (m == points && (n == 1 || n == 3));
    end
    
    function okay = checkColormap(cmap)
        [m,n] = size(cmap);
        okay = (isemptsy(cmap)) || ...
               (m == points && n == 3);
    end
end

function v = computeColorFromZ(pts, n)
    v = pts(:,3);
    v = uint32(round((v - min(v)) / (max(v) - min(v)) * n - 1)) + 1;
end
