% DP2MOT implements a scoped Visual Odometry algorithm based on direct intensity
% matching or optionally a feature-based descriptor matching technique.
%
% See also: epnp
function [mot,x2d] = dp2mot(dp,im,cam0,cam1,varargin)
    % some preliminary processing..
    dp(find(dp < 0 | ~isfinite(dp))) = 0;
    dp = double(dp);
    im = double(im);

    % argument parsing
    po = parseArgs(dp,im,cam0,cam1,varargin{:});

    % optional video output
    if ~isempty(po.Output)
        avi = vision.VideoFileWriter(po.Output,'FrameRate',20);
    end

    % initialise transformation matrices for camera motion
    frames = size(im,3);
    mot = repmat(eye(4),1,1,frames);

    % camera matrix of the left camera..
    % useful for essential-matrix-based outlier rejection
    K = cam0.P * inv(cam0.E);
    K = K(1:3,1:3);

    % ternary if-then-else operator
    ifel = @(varargin) varargin{length(varargin)-varargin{1}};

    % flag to indicate if feature extraction and matching will be required
    useFeatures   = any(strcmpi(po.Method,{'hybrid','indirect'}));
    useDenseImage = any(strcmpi(po.Method,{'hybrid','direct'}));

    % optimisation options
    opts = optimoptions(                 ...
      'lsqnonlin',                       ...
      'Display',   'Iter',               ...
      'Algorithm', 'Levenberg-Marquardt',...
      'MaxIter',   200                   ...
    );

    % iterate thru frames..
    fprintf('Start camera motion recovery..');
    for j = 1 : frames - 1
        ti = ifel(po.RelativeMotion,j,1);
        tj = j + 1;

        % swap frames for backward-mapping
        if po.UseBackwardModel, [tj,ti] = deal(ti,tj); end

        % intensity images
        Ii = im(:,:,ti);
        Ij = im(:,:,tj);

        % convert disparity to a dense 3D point matrix
        Gi = dp2pts(dp(:,:,ti),cam0,cam1);
        Xi = Gi(:,:,1);
        Yi = Gi(:,:,2);
        Zi = Gi(:,:,3);
        Mi = isfinite(Zi) & Zi > 0; % pixels with valid depth data

        % try bootstrapping using optical flow
        if po.UseBootstraping
            Mij = bootstrap(Ii,Ij,K,Gi,po);
        else
            Mij = eye(4);
        end

        % perform feature-based alignment
        if useFeatures
            [src,dst,Mij] = featureFlow(Ii,Ij,K,Mij,po);
            
            if isempty(src),
              warning 'the bootstrapped solution is dishonoured';
              [src,dst,Mij] = featureFlow(Ii,Ij,K,[],po);
            end
            
            ind = sub2ind(size(Ii),round(src(:,2)),round(src(:,1)));
            idx = find(Mi(ind));
            ind = ind(idx);
            src = double([Xi(ind),Yi(ind),Zi(ind)]);
            dst = double(dst(idx,:)) - 1; % minus one for zero-based image coordinates
            Mij = vec2mot(lsqnonlin(@(x)rpe(x,src,dst),mot2vec(Mij,po.UseAngleAxis),[],[],opts),po.UseAngleAxis);
        end

        if useDenseImage
            idx = find(Mi);
            % rnd = randperm(numel(idx));
            % idx = idx(rnd(1:round(numel(idx)*0.25)));
            src = [Xi(idx),Yi(idx),Zi(idx)];
            dst = double(Ii(idx));
            Mij = vec2mot(lsqnonlin(@(x)pme(x,src,dst,Ij),mot2vec(Mij,po.UseAngleAxis),[],[],opts),po.UseAngleAxis);
    % x0   = mot2vec(bootstrap(I0,I1,src.K,pts,initLevel,initCapacity),useAngleAxis);
    % [x,resnorm,residual,exitflag,output,lambda,J] = lsqnonlin(@f,x0,[],[],opts);

    % mot = vec2mot(x,useAngleAxis);
    % x2d = homo2eucl(eucl2eucl(x3d,vec2mot(x,useAngleAxis))*src.K');
    % x2d = list2pack(x2d,idx,size(dp));
        end

        mot(:,:,j+1) = ifel(po.UseBackwardModel,invmot(Mij),Mij);
        fprintf('..%d',j);
    end
    fprintf('..DONE\n');

    if po.RelativeMotion
        for j = 3 : frames
            mot(:,:,j) = mot(:,:,j) * mot(:,:,j-1);
        end
    end

    % reprojection error function
    function y = rpe(x,src,dst)
        y = dst - homo2eucl(eucl2eucl(src,vec2mot(x,po.UseAngleAxis))*K');
        y = reshape(y,[],1);
    end

    % photometric error function
    function y = pme(x,src,dst,Ij)
        src = homo2eucl(eucl2eucl(src,vec2mot(x,po.UseAngleAxis))*K');
        src = interp2(Ij,src(:,1)+1,src(:,2)+1,'linear');
        y = dst - double(src);
        y(find(isnan(y))) = 0;
    end

    return;
    
    function stop = report(x,val,state)
        if exist('avi')
            mot = vec2mot(x,useAngleAxis);
            x2d = homo2eucl(eucl2eucl(x3d,mot)*src.K');
            I10 = zeros(size(I0),'uint8');
            I01 = zeros(size(I0),'uint8');
            
            u = round(x2d(:,1))+1;
            v = round(x2d(:,2))+1;
            
            k = find(u > 0 & u <= size(I0,2) & v > 0 & v <= size(I0,1));
            dst = sub2ind(size(I0),v(k),u(k));
            
            I10(idx) = uint8(interp2(I1,x2d(:,1)+1,x2d(:,2)+1,'cubic'));
            I01(dst) = I0(idx(k));
            
            bad = find(isnan(I10));
            I10(bad) = I0(bad);

            t = mot(1:3,4);
            tx = sprintf('Iteration: %03d / x: %.03fm y: %.03fm z: %.03fm',val.iteration,t/100);
            % im = imblend(uint8(I0),I10);
            im = [imblend(uint8(I0),I10);repmat(I01,1,1,3)];
            im = insertText(im,[32,32],tx,'FontSize',18,'Font','Lucida Console');
            imshow(im); drawnow;
            % if strcmp(state,'init') || strcmp(state,'done')
            %   figure, imshow(im); drawnow;
            % end
            step(avi,im);
        end

        stop = false;
    end
end

function po = parseArgs(dp,im,cam0,cam1,varargin)
    models = {'direct','indirect','hybrid'};

    sizChk = @(x) all(size(x) == size(dp));
    camChk  = @(x) isfield(x,'P') & isfield(x,'E');
    mdlChk  = @(x) any(strcmpi(x,models));

    p = inputParser;
    addRequired(p,'dp',@(x) (ndims(dp) == 2 || ndims(dp) == 3) && isfloat(dp));
    addRequired(p,'im',sizChk);
    addRequired(p,'c0',camChk);
    addRequired(p,'c1',camChk);
    addParameter(p,'Method',models{1},mdlChk);
    addParameter(p,'UseAngleAxis',       false,  @(x) islogical(x));
    addParameter(p,'UseBackwardModel',   true,   @(x) islogical(x));
    addParameter(p,'UseBootstraping',    true,   @(x) islogical(x));
    addParameter(p,'UseBundleAdjustment',false,  @(x) islogical(x));
    addParameter(p,'BootstrapBucket',    0.01,   @(x) isnumeric(x) & x > 0 & x <= 1);
    addParameter(p,'BootstrapLevel',     4,      @(x) isinteger(x) & x > 0);
    addParameter(p,'RelativeMotion',     true,   @(x) islogical(x));
    addParameter(p,'MaxBidirectionalError', 0.5, @(x) isnumeric(x) & x > 0);
    addParameter(p,'MaxEpipolarError',      1.0, @(x) isnumeric(x) & x > 0);
    addParameter(p,'FeatureDetector', @detectSURFFeatures);
    addParameter(p,'Output','',@(x) isempty(x) | all(ischar(x)));

    parse(p,dp,im,cam0,cam1,varargin{:});
    po = p.Results;
end

function mot = bootstrap(I0,I1,K,xyz,po)
    % sub-divide the image domain by m-by-n blocks of h-by-w pixels
    m = 2^po.BootstrapLevel;
    n = m;
    h = ceil(size(I0,1) / m); % block height
    w = ceil(size(I0,2) / n); % block width
    k = ceil(po.BootstrapBucket * h * w); % number of candidates in each block

    % calculate scores
    [X,Y,Z] = slicearray(xyz);
    M = isfinite(Z) & Z > 0;
    S = stdfilt(I0);
    S(find(~M)) = 0;

    % pool of 2D coordinates
    P = zeros(m,n,k,2,'uint16');

    % find best candidates
    for i = 1 : m
        for j = 1 : n
            y0 = (j-1) * h + 1;
            x0 = (i-1) * w + 1;
            y1 = min(y0+h,size(I0,1));
            x1 = min(x0+w,size(I0,2));

            % select k-strongest responses
            [~,idx] = sort(reshape(S(y0:y1,x0:x1),[],1),'descend');
            idx = idx(1:k);
            
            % translate to local pixel coordinates
            [y,x] = ind2sub([y1-y0+1,x1-x0+1],idx);
            
            % translate to global pixel coordinates
            y = y + y0 - 1;
            x = x + x0 - 1;
            
            % store to the pool
            P(i,j,:,:) = reshape([x,y],1,1,k,2);
        end
    end

    % reshape the pool to a 2D point matrix and select points with valid range data
    pts = reshape(P,[],2);
    idx = find(M(sub2ind(size(M),pts(:,2),pts(:,1))));
    src = double(pts(idx,:));

    % optical flow computation
    klt = vision.PointTracker('MaxBidirectionalError',po.MaxBidirectionalError);
    initialize(klt,src,uint8(I0));
    [dst,tracked] = step(klt,uint8(I1));

    % select only tracked points
    idx = find(tracked);
    src = src(idx,:);
    dst = dst(idx,:);

    % figure, imshow(imblend(uint8(I0),uint8(I1))), hold on, plot([src(:,1),dst(:,1)]',[src(:,2),dst(:,2)]','g-');

    % get 3D coordinates of those mapped 2D points
    idx = sub2ind(size(Z),src(:,2),src(:,1));
    src = [X(idx),Y(idx),Z(idx)];

    % figure, plot3(src(:,1),src(:,2),src(:,3),'k.'); axis equal; grid on; pause;

    % motion recovery from the 3D-to-2D correspondences
    % mot = epnp(src,dst,K);
    % use PnP to improve robustness
    optim = optimset('Algorithm','Levenberg-Marquardt');
    solve = @(i) fsolve(@(x) reshape(dst(i,:) - homo2eucl(eucl2eucl(src(i,:),vec2mot(x,po.UseAngleAxis))*K'),[],1), mot2vec(epnp(src(i,:),dst(i,:),K),po.UseAngleAxis), optim);
    feval = @(x) sqrt(sum((dst - homo2eucl(eucl2eucl(src,vec2mot(x,po.UseAngleAxis))*K')).^2,2));
    epsil = po.MaxBidirectionalError;

    [x,rpe,inliers] = ransac(numel(idx),8,...
        solve, ... % solver function
        feval, ... % evaluation function
        epsil, ... % epsilon
        0.95,  ... % confidence
        30,    ... % iterations
        0.50   ... % acceptance ratio
    );

    if isempty(find(inliers)), error 'RANSAC failed'; end
    mot = vec2mot(x,po.UseAngleAxis);
end

function [x0,x1,mot] = featureFlow(I0,I1,K,mot,po)
    if ~isempty(mot)
        priori = ~isequal(mot(1:3,1:3),eye(3)) & norm(mot(1:3,4)) > 0;
    else
        priori = false;
    end

    [x0,f0] = dxtor(uint8(I0));
    [x1,f1] = dxtor(uint8(I1));

    idx = matchFeatures(f0,f1,'Unique',true);
    src = idx(:,1);
    dst = idx(:,2);

    x0 = x0(src,:) - 1;
    x1 = x1(dst,:) - 1;

    if ~isempty(K) && priori
        F = mot2fmat(mot,K,K);
        y = computeEpipolarError(x0,x1,F,'Sampson');
        inliers = find(abs(y) < po.MaxEpipolarError);
    else
        [F,inliers] = estimateFundamentalMatrix(x0,x1,'DistanceThreshold',po.MaxEpipolarError);
        if ~isempty(K)
            K_inv = inv(K);
            E = K'*F*K; % essential matrix from fundamental matrix and camera matrix
            h0 = eucl2homo(x0(inliers,:))*K_inv'; % x0 in homogeneous coordinates
            h1 = eucl2homo(x1(inliers,:))*K_inv'; % x1 in homogeneous coordinates
            mot = emat2mot(E,h0,h1); % motion from essential matrix
        else
            mot = eye(4);
        end
    end

    x0 = x0(inliers,:);
    x1 = x1(inliers,:);

    function [x,f] = dxtor(im)
        [f,x] = extractFeatures(im,po.FeatureDetector(im));
        x = x.Location;
    end
end

function map = list2pack(x2d,idx,sz)
    map = zeros(sz(1),sz(2),size(x2d,2),'like',x2d);
    for k = 1 : size(map,3)
        slice = zeros(sz(1),sz(2),'like',x2d);
        slice(idx) = x2d(:,k);
        map(:,:,k) = slice;
    end
end
