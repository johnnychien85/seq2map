% STX = dp2stx(DP) builds stixel representation STX from disparity map DP
% following Badino et. al's paper "The Stixel World" published in 2009[1]. The
% built stx is stored as an array of structure that comes with feilds x, y,
% width, height, and d (disparity of stixel). When the disparity-depth ratio is
% known, use STX = dp2stx(DP,'FocalLength',F,'BaseLine',B) to get more accurate
% results by supplying F the focal length of the rectified left virtual camera
% and B the length of stereo baseline. See [2] for further information.
%
% STX = dp2stx(DP,PARAM1,VAL1,PARAM2,VAL2,...) builds stixels, specifying
% parameters and corresponding values that decide the construction of them.
%
% General parameters include:
%
%    'StixelWidth'            Width of stixels; set to 5 by default.
%
%    'StixelMinHeight'        Minimum height of stixels. Stixels shorter than
%                             the value will not be retained.
%
%                             The default value is 12.
%
%    'StixelMaxStd'           Maximum standard deviation of disparities of a
%                             stixel. Stixels with deviations above this value
%                             will not be retained.
%
%                             The default value is 2.5.
%
%
%    'StartColumn'            First column to start stixel extraction. By
%                             default the extraction starts from column 1.
%
%    'Ground'                 Ground manifold model. Valid options are 'PLANE'
%                             'LINE', 'POLY', and 'CUT'.
%
%                             The PLANE model performs a RANSAC-based plane
%                             fitting procedure to locate the ground manifold as
%                             a planar surface.
%
%                             The LINE models projects disparities along columns
%                             to the row-disparity space and finds the upper
%                             envelop of the road surface as a line.
%
%                             The POLY model follows a similar way while
%                             modelling the envelop as a polynomial function of
%                             the image's y-axis.
%
%                             The CUT model uses a dynamic-programming technique
%                             to find the envelop by solving a min-cut problem.
%                             See also 'CutSmoothness'.
%
%                             By default PLANE model is adopted.
%
%    'ObstacleThreshold'      The threshold is applied to remove insignificant
%                             cells from an occupancy map. The value is rescaled
%                             by the number of valid disparities in each column.
%                             The disparity corresponding to the cell nearest to
%                             the camera in the thresholded occupancy map will
%                             be selected column-by-column to define the
%                             obstacle manifold.
%
%                             By default the threshold is set to 0.05.
%
%    'ObstacleSmoothness'     The weight controlling smoothness for deciding the
%                             cut separating obstacles from the background in an
%                             occupancy map. The value will be rescaled by the
%                             maximum value of the map.
%
%                             By default it is set to 0.3.
%
%    'TopSmoothness'          The weight controlling smoothness for deciding the
%                             top points for pre-stixels (i.e. column-wise
%                             stixels prior to the grouping/resampling stage)
%
%
%    'FocalLength'            Focal length of the rectified camera pair. Used to
%                             build an accurate membership function for height
%                             segmentation. By default it is set to INF to
%                             approximate the function in disparity space.
%
%    'Baseline'               The baseline of the stereo camera. Used with
%                             'FocalLength' to back-project the disparity map
%                             to build the membership function for height
%                             segmentation in 3D space. By default it is set to
%                             INF to approximate the function in disparity
%                             space.
%
%    'ImageCentre'            Image centre for back-projecting disparities to
%                             the world coordinates for 3D plane-fitting. The
%                             parameter is used only when the 'Ground' is set to
%                             'Plane'.
%
%    'DepthTol'               The width of the Gaussian kernel used to
%                             approximate a binary membership function.
%                             When 'FocalLength' and 'Baseline' are supplied,
%                             the unit is specified in world unit; otherwise in
%                             disparities (i.e. pixels).
%
%                             By default it is set to 2.5.
%
%    'Quiet'                  To suppress figure generation. By default the
%                             value is set to true. To render nicer figures
%                             please also supply the 'Intensity' parameter.
%
%    'Intensity'              Optional left intensity image of the frame; used
%                             only for visualisation.
%
%    'Weight'                 Weight matrix for the disparity map. By default
%                             all pixels are equally weighted.
%
% Ground model-specific parameters include:
%
%    'Skyline'                The row index of skyline. The v-disparity based
%                             modelling will only process cells starting from
%                             'Skyline' to the last row of map. Results will
%                             be used to extrapolate the road manifold from the
%                             first row to 'Skyline'.
%
%                             By default is is set to the number of rows times
%                             0.55.
%
%    'VDispThreshold'         Threshold to remove insignificant cells in a
%                             normalised v-disparity map.
%
%                             By default it is set to 0.30.
%
%    'PlaneTol'               Threshold to decide if a pixel is in-plane for
%                             plane-fitting based ground manifold modelling.
%
%                             By default the value is set to 1.5.
%
%    'PlaneItr'               Number of iterations for the RANSAC-based plane
%                             fitting process. By default it uses 100 trials.
%
%
%    'LineTol'                Threshold to decide if a pixel is in-line for
%                             line-fitting based ground manifold modelling.
%
%                             By default the value is set to 1.5.
%
%    'LineItr'                Number of iterations for the RANSAC-based line
%                             fitting process. By default it uses 100 trials.
%
%    'PolyDegree'             Degree of the polynomial function used to
%                             approximate the road manifold for the polynomial
%                             fitting process. By defauly a order-2 polynomial
%                             is adopted.
%
%    'CutSmoothness'          The weight controlling smoothness for ground
%                             manifold modelling using the 'CUT' model. The
%                             value will be rescaled by the absolute maximum of
%                             the cost function.
%
%                             By default it is set to 0.05.
%
% Author:
%   Hsiang-Jen (Johnny) Chien (jchien@aut.ac.nz)
%   Centre of Robotics and Vision (CeRV)
%   Auckland University of Technology, New Zealand
%
% References:
%  [1] https://link.springer.com/chapter/10.1007/978-3-642-03798-6_6
%  [2] http://www.6d-vision.com/aktuelle-forschung/stixel-world
%
% See also: dp2stx, stx2xml, xml2stx, stxcmp.
%
function [stx,rd,ob,ub] = dp2stx(dp,varargin)
    % parse arguments
    po = parseArgs(dp,varargin{:});

    % initialise output arguments
    stx = []; % extracted stixels
    rd  = []; % road manifold
    ob  = []; % obstacle manifold

    % The first step is to find the road manifold using a v-disparity graph-cut
    % technique. Note in the Badino's paper B-splines are used to model the cut
    % in the v-disparity space. The road manifold is coded as a disparity map
    % where rd(x,y) denotes the disparity value of the manifold in (x,y).
    [rd,ub] = road(dp,po);

    if any(~isfinite(rd)), return; end

    % The second step is to find base points using an occupancy grid (i.e.
    % u-disparity map).
    [bs,ch,ob] = base(dp,rd,ub,po);

    if isempty(bs), return; end

    % The third step is to find top points using height segmentation.
    tp = top(dp,bs,po);

    % The last step is to extract stixels
    stx = stixel(dp,bs,tp,ch,po);
end

function po = parseArgs(dp,varargin)
    p = inputParser;
    addParameter(p,'StixelWidth',5);
    addParameter(p,'StixelMinHeight',7);
    addParameter(p,'StixelMaxStd',2.5);
    addParameter(p,'StartColumn',1);
    addParameter(p,'Ground','Cut');
    addParameter(p,'ObstacleThreshold',0.05);
    addParameter(p,'ObstacleSmoothness',0.30);
    addParameter(p,'TopSmoothness', 0.01);
    addParameter(p,'Intensity',ones(size(dp),'uint8')*127);
    addParameter(p,'Weight',ones(size(dp)));
    addParameter(p,'FocalLength',Inf);
    addParameter(p,'Baseline',Inf);
    addParameter(p,'ImageCentre',[Inf,Inf]);
    addParameter(p,'DepthTol',2.5);
    addParameter(p,'PlaneTol',1.5);
    addParameter(p,'PlaneItr',100);
    addParameter(p,'LineTol',1.5);
    addParameter(p,'LineItr',100);
    addParameter(p,'VDispThreshold',0.20);
    addParameter(p,'PolyDegree',2);
    addParameter(p,'CutSmoothness', 0.05);
    addParameter(p,'CutDim',2);
    addParameter(p,'Skyline',round(size(dp,1)*0.55));
    addParameter(p,'Quiet',true);
    parse(p,varargin{:});
    po = p.Results;
end

function [R,U] = road(dp,po)
    if ~isa(po.Ground,'function_handle')
        switch lower(po.Ground)
            case 'plane', f = @roadPlaneFit;
            case 'line',  f = @roadLineFit;
            case 'poly',  f = @roadPolyFit;
            case 'cut',   f = @roadGraphCut;
            otherwise error('unknown ground model: %s',po.Ground);
        end
    else
        f = po.Ground;
    end

    [R,U] = f(dp,po);
end

%
% RANSAC-based plane-fitting road manifold in 3D space;
% Requires Computer Vision System Toolbox.
%
function [R,U] = roadPlaneFit(dp,po)
    assert(exist('pcfitplane')==2,'Computer Vision System Toolbox required');

    idx = find(dp > 0);
    [v,u] = ind2sub(size(dp),idx);

    % use only pixels below the skyline
    I = find(v > po.Skyline);
    idx = idx(I);

    % perform back-projection to do plane fitting in the Euclidean space
    bf = po.Baseline * po.FocalLength;
    uc = po.ImageCentre(1);
    vc = po.ImageCentre(2);
    euclidean = all(isfinite(po.ImageCentre)) && isfinite(bf);
    
    % build a point cloud and perform RANSAC plane fitting
    if euclidean
        z = bf ./ dp(idx);
        x = z .* (u(I) - uc) / po.FocalLength;
        y = z .* (v(I) - vc) / po.FocalLength;
        pc = pointCloud([x,y,z],'color',repmat(po.Intensity(idx),1,3));
    else
        pc = pointCloud([u(I),v(I),dp(idx)],'color',repmat(po.Intensity(idx),1,3));
    end

    % plane fitting
    plane = pcfitplane(pc,po.PlaneTol,'MaxNumTrials',po.PlaneItr);

    % coefficients of the found plane
    a = plane.Parameters;

    % make sure the normal vector is in the opposite direction to the
    % camera's y-axis.
    if a(2) > 0, a = -a; end

    % compute the disparity map of road manifold
    [x,y] = meshgrid(1:size(dp,2),1:size(dp,1));

    % upper envelop
    d = a(4) - sign(a(4))*po.PlaneTol;

    if euclidean
        x = (x - uc) / po.FocalLength;
        y = (y - vc) / po.FocalLength;
        z = bf ./ dp;
        Z = -a(4)./(a(1)*x + a(2)*y + a(3));  % depth of the road manifold
        R = bf ./ Z;                          % convert to disparity
        U = -bf * (a(1)*x + a(2)*y + a(3)) ./ d;   % upper envelope
        E = a(1)*x.*z + a(2)*y.*z + a(3)*z + a(4); % elevation map
    else
        R = -(a(1)*x + a(2)*y + a(4))/a(3);   % disparity of the road manifold
        U = -(a(1)*x + a(2)*y + d   )/a(3);   % disparity of the enveope
        E = a(1)*x + a(2)*y + a(3)*dp + a(4); % elevation map
    end

    R(find(R < 0)) = 0;
    U(find(R == 0)) = 0;

    % skip visualisation
    if po.Quiet, return; end

    span = [-po.PlaneTol,po.PlaneTol];
    rgb = uint8(val2rgb(E,jet,span)*255);
    im  = repmat(po.Intensity,1,1,3);
    figure, imshow(rgb*0.5+im*0.5); colormap jet, colorbar; caxis(span);

    figure, pcshow(pc), hold on;
    title 'Road manifold';
    
    if euclidean
        xlabel 'X', ylabel 'Y', zlabel 'Z';
        x = x.*Z;
        y = y.*Z;
        z = Z;
    else
        xlabel 'Column (px)', ylabel 'Row (px)', zlabel 'Disparity (px)';
        z = R;
    end
    
    surf(x(1:10:end,1:10:end),y(1:10:end,1:10:end),z(1:10:end,1:10:end),'EdgeColor','g','FaceColor','white','FaceAlpha',0.5);
end

%
% Road manifold from v-disparity based line-fitting
%
function [R,U] = roadLineFit(dp,po)
    % build v-disparity map and upper envelop-based cost function
    [vd,cst] = dp2vd(dp,po);

    %% implementation using Hough transforms
    % extrapolate the extended road manifold
    % sky = po.Skyline;
    % vd(1:sky,:) = 0;
    %
    % line fitting using a Hough transform
    % bw = edge(vd,'canny');
    % [H,T,R] = hough(bw);
    % P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
    % lines = houghlines(bw,T,R,P);
    % d = lines(1).point2 - lines(1).point1;
    % m = d(1)/d(2);
    % b = lines(1).point1(1) - m*lines(1).point1(2);
    % cut = m*(1:size(dp,1))'+b;
    
    % line fitting using RANSAC
    % cst(find(vd < po.VDispThreshold*size(vd,2))) = 0;

    cst(1:po.Skyline,:) = 0;
    cst(find(vd < po.VDispThreshold)) = 0;
    cut = rlinefit(cst)';

    % back-project the cut to image domain    
    R = repmat(cut,1,size(dp,2));
    R(find(R<0)) = 0;

    %...
    U = R + po.LineTol;
    U(find(R==0)) = 0;
    %...

    % visualisation
    if ~po.Quiet, vdshow(vd,cut); end

    function f = rlinefit(A)
        [y,x,v] = find(A);
        n = numel(y);

        f_bst = zeros(3,1);
        e_bst = inf;

        for k = 1 : po.LineItr
            p0 = round(rand*(n-1))+1;
            p1 = round(rand*(n-1))+1;

            x0 = x(p0);
            y0 = y(p0);
            x1 = x(p1);
            y1 = y(p1);

            mk = (y1-y0)/(x1-x0);
            bk = y0 - mk*x0;
            ak = [mk,-1,bk];
            
            ak = ak ./ norm(ak(1:2));
            if ak(3) > 0, ak = -ak; end

            % ek = sum(((ak(1)*x + ak(2)*y + ak(3)) .* v).^2);
            inliers = find(abs(ak(1)*x + ak(2)*y + ak(3)) < po.LineTol);
            ek = sum(v(inliers));

            if ek < e_bst
                e_bst = ek;
                f_bst = ak;
            end
        end

        m = -f_bst(2)/f_bst(1);
        b = -f_bst(3)/f_bst(1);
        f = m*[1:size(A,1)] + b;
    end    
end

%
% Road manifold v-disparity based polynomial-fitting
%
function [R,U] = roadPolyFit(dp,po)
    % build v-disparity map and find the upper envelop
    [vd,cst] = dp2vd(dp,po);
    y = [1:size(vd,1)]';
    [~,x] = min(cst,[],2);
    
    idx = po.Skyline:numel(y);
    idx = idx(find(x(idx) > 1));
    ind = sub2ind(size(dp),y(idx),x(idx));
    idx = idx(find(vd(ind) > po.VDispThreshold));
    
    poly = polyfit(y(idx),x(idx),po.PolyDegree);
    cut  = polyval(poly,y);

    % back-project the cut to image domain    
    R = repmat(cut,1,size(dp,2));
    R(find(R<0)) = 0;

    %...
    U = R + po.LineTol;
    U(find(R==0)) = 0;
    %...

    % visualisation
    if ~po.Quiet,
        vdshow(vd,cut); hold on;
        % plot(x(idx),y(idx),'go-');
    end
end

%
% Road manifold from v-disparity based graph-cut.
%
function [R,U] = roadGraphCut(dp,po)
    % build v-disparity map and find the upper envelop
    [vd,cst] = dp2vd(dp,po,true);

    cst = cst(po.Skyline+1:end,:);
    tol = po.CutSmoothness * max(abs(cst(:)));

    % Model 1: vertical cut
    if po.CutDim == 1
        % solve for the cut using dynamic programming
        cut = bisect2(cst',tol,'+')';

        % extrapolate the extended road manifold
        cut = [zeros(po.Skyline,1);double(cut)];
        sky = po.Skyline;
        y0  = sky+1;
        y1  = numel(cut);
        cut(1:sky) = ([1:sky]-y0)/(y1-y0)*(cut(y1)-cut(y0)) + cut(y0);
    else
    % Model 2: horizontal cut + interpolation
        % solve for the cut using dynamic programming
        cut = bisect2(cst,tol,'+'); % this gives a disparity-row cut
        
        % get row-disparity cut by inverse-interpolation
        x0 = max(find(cut == min(cut)));
        x1 = min(find(cut == max(cut)));
        x  = x0:x1;
        y  = double(cut(x)) + po.Skyline;

        [y,i] = unique(y);
        cut = interp1(y,x(i),1:size(dp,1),'linear','extrap')';
    end

    % back-project the cut to image domain    
    R = repmat(cut,1,size(dp,2));
    R(find(R<0)) = 0;

    %...
    U = R + po.LineTol;
    U(find(R==0)) = 0;
    %...

    % visualisation
    if ~po.Quiet, vdshow(vd,cut); end
end

%
% Convert a disparity to v-disparity map
%
function [vd,cst] = dp2vd(dp,po,grady)
    if nargin < 3, grady = false; end

    % find valid pixels
    idx = find(dp > 0);
    [v,u] = ind2sub(size(dp),idx);

    % build a v-disparity map from the quantised disparity map    
    dp = round(dp)+1; % quantisation
    vd = accumarray([v,dp(idx)],po.Weight(idx),[size(dp,1),max(dp(idx))]); % build a histogram

    % normalisation
    vd = bsxfun(@rdivide,vd,sum(vd,2));

    %...
    cst = -vd;
    return;
    %...

    % to find a minimum cut thru the greatest rising edge of the histogram
    [P,Q] = gradient(vd);

    if grady, cst = -Q;
    else      cst = P;
    end
end

%
% Visualise a v-disparity map with the upper envelop
%
function vdshow(vd,cut)
    figure, imagesc(vd), colormap(flipdim(gray,1)), hold on;
    plot(cut(1:1:end),1:1:numel(cut),'rs-','LineWidth',1,'MarkerSize',5,'MarkerFaceColor','r');
    title 'v-disparity', xlabel 'Disparity (px)', ylabel 'Row (px)';
end

%
% Detect base-points from ground and obstacle manifolds
%
function [cut,chk,B] = base(dp,R,U,po)
    % keep only disparities above the road manifold
    % (i.e. pixels closer to the camera compared to the manifold)
    idx = find(dp > U);
    [v,u] = ind2sub(size(dp),idx);

    if isempty(idx)
        cut = [];
        B = [];
        return;
    end

    % build a polar-occupancy grid (i.e. u-disparity map)
    dp = round(dp)+1; % quantisation
    ud = accumarray([dp(idx),u],po.Weight(idx),[max(dp(idx)),size(dp,2)]); % histograming
    %    accumarray([dp(idx),u],ones(numel(idx),1),[max(dp(idx)),size(dp,2)]); % histograming
    ud(1,:) = 0; % remove zero disparities

    % background removal
    ub = zeros(size(ud),'like',ud);
    for j = 1 : size(ud,2)
        cutoff = po.ObstacleThreshold * sum(ud(:,j));
        % i = max(find(ud(:,j) > cutoff));
        ud(find(ud(:,j) < cutoff),j) = 0;
        i = max(find(ud(:,j) > 0));
        ub(i:end,j) = ud(i:end,j);
    end

    cst = -double(ub);
    tol = po.ObstacleSmoothness * max(ub(:));
    cut = bisect2(cst,tol);
    chk = ub(sub2ind(size(ub),cut,1:numel(cut)));
    B = double(repmat(cut-1,size(dp,1),1));

    % visualisation
    if ~po.Quiet
        figure;
        subplot(1,2,1), imagesc(ud), title 'Polar occupancy grid', xlabel 'Column (px)', ylabel 'Disparity (px)', colormap 'hot';
        subplot(1,2,2), imagesc(ub), title 'Background removed',   xlabel 'Column (px)', ylabel 'Disparity (px)', colormap 'hot';
        hold on; plot(1:numel(cut),cut,'g-');
    end

    cst = abs(B-R);
    [~,cut] = min(cst,[],1);

    % D = abs(dp - R) < 1;
    % chk(find(~D(sub2ind(size(dp),cut,1:numel(cut))))) = Inf;

    % skip visualisation
    if po.Quiet, return; end

    % figure, imagesc(cst), colormap 'hot', title 'Base cost', xlabel 'Column (px)', ylabel 'Row (px)';
    rgb = uint8(255*val2rgb(cst,jet(255),[0,10]));
    im  = repmat(po.Intensity,1,1,3);
    figure, imshow(rgb*0.4+im*0.6), title 'Base-point cost', xlabel 'Column (px)', ylabel 'Row (px)';
    hold on, plot(1:numel(cut),cut,'g-','LineWidth',3);

    figure;
    pcshow(pointCloud([u,v,dp(idx)-1],'color',repmat(im(idx),1,3)),'MarkerSize',3); hold on;
    % mesh(medfilt2(dp)); colormap 'gray';  hold on;
    surf(R,'EdgeColor','c','FaceColor','white','FaceAlpha',0.5);
    surf(B,'EdgeColor','g','FaceColor','white','FaceAlpha',0.5);
    axis equal;
end

%
% Detect top-points by means of a benefit image.
%
function cut = top(dp,bs,po)
    bf = po.FocalLength * po.Baseline;
    dz = po.DepthTol;

    % backproject the disparity of base points to generate a disparity map
    % (equivelant to Du in Badino's paper)
    du = repmat(dp(sub2ind(size(dp),bs,1:numel(bs))),size(dp,1),1);

    % generate a mask of pixels above base points
    M = zeros(size(dp),'logical');
    for j = 1 : size(dp,2), M(1:bs(j)-1,j) = 1; end

    % compute memebership function (equivelant to M_{u,v} in the paper)
    if bf > 0
        D = du - bf./(bf./du+dz); % depth-based window size (see Eq. 2)
        E = 2.^(1-((dp-du)./D).^2)-1; % membership function (see Eq. 1)
    else
        E = 2.^(1-((dp-du)./dz).^2)-1; % disparity-based window size
    end

    % integrate the membership votes to get a cost function
    E(find(~M | isnan(E))) = 0;

    Ef = cumsum(E,1,'forward'); % forward integral
    Eb = cumsum(E,1,'reverse'); % backward integral
    cst = Ef - Eb + E; % cost image (see Eq. 3)

    % find the minimum cut to decide top points (note depth discontinuities are
    % not taken into account to model the smoothness term; check Eq. 5)
    cst(find(~M)) = Inf;
    idx = find(isfinite(cst));
    tol = po.TopSmoothness * max(abs(cst(idx)));
    cut = bisect2(cst,tol);

    % skip visualisation
    if po.Quiet, return; end

    figure, imagesc(E), colormap 'gray', title 'Membership value', xlabel 'Column (px)', ylabel 'Row (px)';
    % figure, imagesc(cst), colormap 'hot', title 'Top cost', xlabel 'Column (px)', ylabel 'Row (px)';
    rgb = uint8(255*val2rgb(cst,jet(255),[min(cst(idx)),max(cst(idx))]));
    im  = repmat(po.Intensity,1,1,3);
    figure, imshow(rgb*0.4+im*0.6), title 'Top-point cost', xlabel 'Column (px)', ylabel 'Row (px)';
    hold on;
    idx = po.StartColumn:numel(cut);
    plot(idx,cut(idx),'g-','LineWidth',3);
    plot(idx,bs(idx), 'w-','LineWidth',2);
end

%
% Stixel extraction.
%
function stx = stixel(dp,bs,tp,ch,po)
    sw = po.StixelWidth;
    stx = [];

    for j = 1 : numel(bs)
        if isfinite(bs(j)) & sum(po.Weight(tp(j):bs(j),j)) < ch(j)
            bs(j) = inf;
        end
    end

    % grouping top and base points to build stixels
    for j = po.StartColumn : sw : size(dp,2)
        x0 = j;
        x1 = min(j+sw-1,size(dp,2));
        y0 = min(tp(x0:x1));
        y1 = max(bs(x0:x1));
        dj = extract(x0,x1); % reshape(dp(y0:y1,x0:x1),1,[]);
        d0 = min(dj);
        d1 = max(dj);

        if ~isempty(dj) &  std(dj) < po.StixelMaxStd & (y1 - y0) > po.StixelMinHeight & y1 > po.Skyline
            n = numel(stx)+1;
            stx(n).x = x0;
            stx(n).y = y0;
            stx(n).d = mean(dj);
            stx(n).width  = x1 - x0 + 1;
            stx(n).height = y1 - y0 + 1;
            stx(n).dspan = [d0,d1];
        end
    end

    % skip visualization
    if po.Quiet, return; end

    figure, imshow(dp,[]), title 'Base (green) and Top (red)', hold on;
    idx = po.StartColumn:numel(tp);
    plot(idx,bs(idx),'g-',idx,tp(idx),'r-');

    function dj = extract(x0,x1)
        if any(~isfinite(bs(x0:x1)))
            dj = [];
            return;
        end
    
        dj = cell(x1-x0+1,1);
        for x = x0 : x1
            dj{x-x0+1} = dp(tp(x):bs(x),x);
        end
        dj = cat(1,dj{:});
        dj = dj(find(dj > 0 & isfinite(dj)));
    end
end
