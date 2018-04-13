% DP2EL identifies road manifold using DP2STX in the first frame and uses it to
% build a digital elevation map (DEM) by accumulating transformed 3D points for
% each disparity map.
function [dem,z,x] = dp2el(dp,cam0,cam1,mot,weight)
    frames = size(dp,3);
    assert(frames == size(mot,3));

    if nargin > 4
        assert(size(weight,3) == frames);
        assert(size(weight,1) == size(dp,1) & size(weight,2) == size(dp,2));
    else
        weight = [];
    end

    t0 = 1;
    zrange = [0,15]; % scan potholes up to 60m
    xrange = [-5,5]; % define the width of corrider
    res = 0.05;      % cell resolution set to 0.1m

    fprintf('Finding road manifold..');
    tfm = findRoadTransform(dp(:,:,t0),cam0,cam1);
    fprintf('..DONE\n');

    % accumulate frames
    fprintf('Accumulating depth data..');
    for t = 1 : frames
        % compute the transform from frame t to RCS
        Mt = tfm * mot(:,:,t0) * invmot(mot(:,:,t));

        % back-project points to 3D space and apply the transform
        [x,y,z] = slicearray(eucl2eucl(dp2pts(dp(:,:,1),cam0,cam1),Mt));

        % find points within the range of interest
        if isempty(weight)
            idx = find(x > xrange(1) & x < xrange(2) & z > zrange(1) & z < zrange(2));
            G{t} = [x(idx),y(idx),z(idx)];
        else
            w = weight(:,:,t);
            idx = find(x > xrange(1) & x < xrange(2) & z > zrange(1) & z < zrange(2) & w > 0);
            G{t} = [x(idx),y(idx),z(idx)];
            W{t} = w(idx);
        end

        fprintf('..%d',t);
    end
    fprintf('..DONE\n');

    [x,y,z] = slicearray(cat(1,G{:}));

    % transform from the road-centered coordinate system to elevation map
    i  = round((z-zrange(1))/res)+1;
    j  = round((x-xrange(1))/res)+1;
    sz = round([(zrange(2)-zrange(1))/res,(xrange(2)-xrange(1))/res])+1;
    
    if isempty(weight)
        dem = -accumarray([i,j],y,sz,@mean,NaN);
    else
        w = cat(1,W{:});
        dem = -accumarray([i,j],y.*w,sz,@sum,NaN)./accumarray([i,j],w,sz,@sum);
    end

    z = zrange(1):res:zrange(2);
    x = xrange(1):res:xrange(2);
end

function tfm = findRoadTransform(dp,cam0,cam1)
    E = cam1.E * invmot(cam0.E);
    baseline = -E(1,4);
    focalLength = cam0.K(1,1);

    [~,rd,~,ub] = dp2stx(dp,     ...
      'FocalLength',focalLength,...
      'Baseline',   baseline,   ...
      'Quiet',      true,       ...
      'Ground',     'Plane',    ...
      'PlaneTol',   0.5,        ...
      'PlaneItr',   500,        ...
      'SkyLine',    128);
    
    [x,y,z] = slicearray(dp2pts(dp,cam0,cam1));

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
    tfm = [0,1,0;0,0,1;,1,0,1]*[R,t];

    % preserve the shift in depth
    tfm(3,4) = 0;
end
