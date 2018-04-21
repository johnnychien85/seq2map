% DP2EL identifies road manifold using DP2STX in the first frame and uses it to
% build a digital elevation map (DEM) by accumulating transformed 3D points for
% each disparity map.
function [el,zgrid,xgrid,rcs] = dp2el(dp,cam0,cam1,mot,varargin)
    po = parseArgs(dp,cam0,cam1,mot,varargin{:});
    frames = size(dp,3);

    t0 = 1;

    if isempty(po.RCS)
        fprintf('Finding road manifold..');
        rcs = findRoadTransform(dp(:,:,t0),po);
        fprintf('..DONE\n');
    else
        rcs = po.RCS;
    end

    % accumulate frames
    fprintf('Accumulating depth data..');
    for t = 1 : frames
        % compute the transform from frame t to RCS
        Mt = rcs * mot(:,:,t0) * invmot(mot(:,:,t));

        % back-project points to 3D space and apply the transform
        [x,y,z] = slicearray(eucl2eucl(dp2pts(dp(:,:,t),cam0,cam1),Mt));

        % find points within the range of interest
        if isempty(po.Weight)
            idx = find(x > po.XLim(1) & x < po.XLim(2) & z > po.ZLim(1) & z < po.ZLim(2));
            G{t} = [x(idx),y(idx),z(idx)];
        else
            w = po.Weight(:,:,t);
            idx = find(x > po.XLim(1) & x < po.XLim(2) & z > po.ZLim(1) & z < po.ZLim(2) & w > 0);
            G{t} = [x(idx),y(idx),z(idx)];
            W{t} = w(idx);
        end

        % I0 = im(:,:,t);
        % pcwrite(pointCloud(G{t},'color',repmat(I0(idx),1,3)),sprintf('pc%02d.ply',t));

        fprintf('..%d',t);
    end
    fprintf('..DONE\n');

    [x,y,z] = slicearray(cat(1,G{:}));

    % transform from the road-centered coordinate system to elevation map
    i  = round((z-po.ZLim(1))/po.ZRes)+1;
    j  = round((x-po.XLim(1))/po.XRes)+1;
    sz = round([(po.ZLim(2)-po.ZLim(1))/po.ZRes,(po.XLim(2)-po.XLim(1))/po.XRes])+1;
    
    if ~exist('W')
        el = -accumarray([i,j],y,sz,@mean,-Inf);
    else
        w  = cat(1,W{:});
        n  = accumarray([i,j],w,sz,@sum);
        el = -accumarray([i,j],y.*w,sz,@sum,-Inf)./n;
    end

    zgrid = po.ZLim(1):po.ZRes:po.ZLim(2);
    xgrid = po.XLim(1):po.XRes:po.XLim(2);
end

function po = parseArgs(dp,cam0,cam1,mot,varargin)
    camChk = @(x) isfield(x,'P') && isfield(x,'E') && isfield(x,'K');
    limChk = @(x) numel(x) == 2 && x(1) < x(2);
    resChk = @(x) isfinite(x) && x > 0;
    sizChk = @(x) isempty(x) || all(size(x) == size(dp));
    tfmChk = @(x) isempty(x) || ismot(x);

    p = inputParser;
    addRequired(p,'dp',@(x)(ndims(dp) == 2 || ndims(dp) == 3) && isfloat(dp));
    addRequired(p,'c0',camChk);
    addRequired(p,'c1',camChk);
    addRequired(p,'mot',@(x) size(mot,3) == size(dp,3));
    addParameter(p,'ZLim',[5,10],limChk);
    addParameter(p,'XLim',[-5,5],limChk);
    addParameter(p,'ZRes',0.01,  resChk);
    addParameter(p,'XRes',0.01,  resChk);
    addParameter(p,'Weight',[],  sizChk);
    addParameter(p,'RCS',[],     tfmChk);
    
    parse(p,dp,cam0,cam1,mot,varargin{:});
    po = p.Results;
end

function rcs = findRoadTransform(dp,po)
    E = po.c1.E * invmot(po.c0.E);
    baseline = -E(1,4);
    focalLength = po.c0.K(1,1);

    [~,rd,~,ub] = dp2stx(dp,    ...
      'FocalLength',focalLength,...
      'Baseline',   baseline,   ...
      'Quiet',      true,       ...
      'Ground',     'Plane',    ...
      'PlaneTol',   0.5,        ...
      'PlaneItr',   500,        ...
      'SkyLine',    128         ...
    );
    
    [x,y,z] = slicearray(dp2pts(dp,po.c0,po.c1));

    % idx = find(dp > 0 & dp < ub);
    % idx = find(abs(dp-rd) < 1 & dp > 0);
    idx = find(dp <= ub & dp > max(ub-1,0));
    G = [x(idx),y(idx),z(idx)];
    R = pca(G)';

    % Force rotations to be proper, i. e. det(R) = 1
    if det(R) < 0, R = -R; end

    % zero-mean shift
    t = -R*mean(G)';

    % make z-axis the first principal axis
    rcs = [0,1,0;0,0,1;,1,0,0]*[R,t];

    % preserve the shift in depth
    rcs(3,4) = 0;
end
