function [ld,idx] = loadLiDARData(lid, frame, varargin)
    po = parseArgs(varargin{:});

    switch lid.Format
        case {'VLD64', 'VLD32', 'VLP16'}
            [ld.data,ld.theta] = VLDLoadData(lid.DataFiles{frame}, lid.Format);
            [~,ld.reflx] = VLDData2RangeReflx(ld.data);
            [ld.points,ld.valid] = VLDData2Points(ld.data, ld.theta, lid.AT);
            if po.ReshapePoints
                ld.points = reshape(ld.points, [], 3);
                ld.reflx  = reshape(ld.reflx,  [], 1);
                ld.valid  = reshape(ld.valid,  [], 1);
            end
        case 'XYZI'
            ld.data   = loadXYZI(lid.DataFiles{frame});
            ld.points = ld.data(:,1:3);
            ld.reflx  = ld.data(:,4);
            ld.valid  = true(size(ld.points,1),1);
        otherwise error 'unknown LiDAR model';
    end
    
    if po.ValidPointsOnly
        idx = find(ld.valid);
        ld.points = ld.points(idx,:);
        ld.reflx = ld.reflx(idx,:);
        ld.valid = ld.valid(idx);
    else
        idx = reshape(1:numel(ld.valid),size(ld.valid));
    end
    
    if po.ApplyTransform
        ld.points = eucl2eucl(ld.points,lid.E);
    end
end

function po = parseArgs(varargin)
    po = inputParser;
    addParameter(po, 'ReshapePoints',    true);
    addParameter(po, 'ValidPointsOnly', false);
    addParameter(po, 'ApplyTransform',  false);

    parse(po, varargin{:});
    po = po.Results;
end

function x = loadXYZI(filename)
    d = dir(filename);
    m = 4;
    n = d.bytes / 4;

    assert(mod(d.bytes, m) == 0);
    f = fopen(filename, 'r');
    x = fread(f, [m,n], 'single')';
    fclose(f);
end
