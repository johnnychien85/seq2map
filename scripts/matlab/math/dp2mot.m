% DP2MOT implements a scoped Visual Odometry algorithm based on direct intensity
% matching or optionally a feature-based descriptor matching technique.
function [mot,x2d] = dp2mot(dp,im,src,dst,varargin)
    % some preliminary processing..
    dp(find(dp < 0 | ~isfinite(dp))) = 0;
    dp = double(dp);
    im = double(im);

    % argument parsing
    po = parseArgs(dp,im,src,dst,varargin{:});

    % optional video output
    if ~isempty(po.Output)
        avi = vision.VideoFileWriter(po.Output,'FrameRate',20);
    end

    % initialise transformation matrices for camera motion
    frames = size(im,3);
    mot = repmat(eye(4),1,1,frames);

    % camera matrix of the left camera..
    % useful for essential-matrix-based outlier rejection
    K = src.P * inv(src.E);
    K = K(1:3,1:3);

    % ternary if-then-else operator
    ifel = @(varargin) varargin{length(varargin)-varargin{1}};

    % flag to indicate if feature extraction and matching will be required
    useFeatures   = any(strcmpi(po.Method,{'hybrid','indirect'}));
    useDenseImage = any(strcmpi(po.Method,{'hybrid','direct'}));

    % optimisation options
    opts = optimoptions('lsqnonlin','Display','None','Algorithm','Levenberg-Marquardt','MaxIter',200);

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
        Gi = dp2pts(dp(:,:,ti),src,dst);
        Xi = Gi(:,:,1);
        Yi = Gi(:,:,2);
        Zi = Gi(:,:,3);
        Mi = isfinite(Zi);

        % try bootstrapping using optical flow
        if po.UseBootstraping
            Mij = bootstrap(Ii,Ij,K,Gi,po);
        else
            Mij = eye(4);
        end

        if useFeatures
            [xi,xj,Mij] = findCorrespondences(Ii,Ij,K,Mij,po);
            idx = sub2ind(size(Ii),round(xi(:,2)),round(xi(:,1)));
            I  = find(Mi(idx));
            xi = [Xi(idx(I)),Yi(idx(I)),Zi(idx(I))];
            xj = xj(I,:);

            f_rpe = @(x) rpe(x,double(xi),double(xj-1));
            Mij = vec2mot(lsqnonlin(f_rpe,mot2vec(Mij,po.UseAngleAxis),[],[],opts),po.UseAngleAxis);
        end

        if useDenseImage
            
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

    function y = rpe(x,src,dst)
        y = dst - homo2eucl(eucl2eucl(src,vec2mot(x,po.UseAngleAxis)) * K');
        y = reshape(y,[],1);
    end
    
    return;
    
    x = pts(:,:,1);
    y = pts(:,:,2);
    z = pts(:,:,3);

    idx = find(isfinite(z));
    x3d = [x(idx),y(idx),z(idx)];

    % nonlinear adjustment
    opts = optimoptions('lsqnonlin','Display','iter','Algorithm','levenberg-marquardt','MaxIter',200,'OutputFcn',@report);
    x0   = mot2vec(bootstrap(I0,I1,src.K,pts,initLevel,initCapacity),useAngleAxis);
    [x,resnorm,residual,exitflag,output,lambda,J] = lsqnonlin(@f,x0,[],[],opts);

    mot = vec2mot(x,useAngleAxis);
    x2d = homo2eucl(eucl2eucl(x3d,vec2mot(x,useAngleAxis))*src.K');
    x2d = list2pack(x2d,idx,size(dp));

    function y = f(x)
        x2d = homo2eucl(eucl2eucl(x3d,vec2mot(x,useAngleAxis))*src.K');
        y = I0(idx) - interp2(I1,x2d(:,1)+1,x2d(:,2)+1,'linear');

        y(find(isnan(y))) = 0;
    end

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

function po = parseArgs(dp,im,src,dst,varargin)
    models = {'direct','indirect','hybrid'};

    sizeChk = @(x) all(size(x) == size(dp));
    camChk  = @(x) isfield(x,'P') & isfield(x,'E');
    mdlChk  = @(x) any(strcmpi(x,models));

    p = inputParser;
    addRequired(p,'dp',@(x) (ndims(dp) == 2 || ndims(dp) == 3) && isfloat(dp));
    addRequired(p,'im',sizeChk);
    addRequired(p,'c0',camChk);
    addRequired(p,'c1',camChk);
    addParameter(p,'Method',models{1},mdlChk);
    addParameter(p,'UseAngleAxis',       false,  @(x) islogical(x));
    addParameter(p,'UseBackwardModel',   false,  @(x) islogical(x));
    addParameter(p,'UseBootstraping',    true,   @(x) islogical(x));
    addParameter(p,'UseBundleAdjustment',false,  @(x) islogical(x));
    addParameter(p,'BootstrapBucket',    0.01,   @(x) isnumeric(x) & x > 0 & x <= 1);
    addParameter(p,'BootstrapLevel',     4,      @(x) isinteger(x) & x > 0);
    addParameter(p,'RelativeMotion',     true,   @(x) islogical(x));
    addParameter(p,'MaxBidirectionalError', 0.5, @(x) isnumeric(x) & x > 0);
    addParameter(p,'MaxEpipolarError',      1.0, @(x) isnumeric(x) & x > 0);
    addParameter(p,'FeatureDetector', @detectSURFFeatures);
    addParameter(p,'Output','',@(x) isempty(x) | all(ischar(x)));

    parse(p,dp,im,src,dst,varargin{:});
    po = p.Results;
end

function mot = bootstrap(I0,I1,K,xyz,po)
    % sub-divide the image domain by m-by-n h-by-w blocks
    m = 2^po.BootstrapLevel;
    n = m;
    h = ceil(size(I0,1) / m); % block height
    w = ceil(size(I0,2) / n); % block width
    k = ceil(po.BootstrapBucket * h * w); % number of candidates in each block

    % calculate scores
    X = xyz(:,:,1);
    Y = xyz(:,:,2);
    Z = xyz(:,:,3);
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

    pts = reshape(P,[],2);
    idx = find(M(sub2ind(size(M),pts(:,2),pts(:,1))));
    src = double(pts(idx,:));

    klt = vision.PointTracker('MaxBidirectionalError',po.MaxBidirectionalError);
    initialize(klt,src,uint8(I0));
    [dst,tracked] = step(klt,uint8(I1));

    idx = find(tracked);
    src = src(idx,:);
    dst = dst(idx,:);

    figure, imshow(imblend(uint8(I0),uint8(I1))), hold on, plot([src(:,1),dst(:,1)]',[src(:,2),dst(:,2)]','g-');
    
    idx = sub2ind(size(Z),src(:,2),src(:,1));
    src = [X(idx),Y(idx),Z(idx)];

    % figure, plot3(src(:,1),src(:,2),src(:,3),'k.'); axis equal; grid on; pause;

    mot = epnp(src,dst,K);

    % rpe = dst - homo2eucl(eucl2eucl(src,mot)*K');
    % figure, plot(rpe(:,1),rpe(:,2),'b.');
end

function [x0,x1,mot] = findCorrespondences(I0,I1,K,mot,po)
    priori = ~isempty(mot) & ~isequal(mot(1:3,1:3),eye(3)) & norm(mot(1:3,4)) > 0;

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
    elseif priori
        [F,inliers] = estimateFundamentalMatrix(x0,x1,'DistanceThreshold',po.MaxEpipolarError);
        K_inv = inv(K);
        E = K'*F*K; % essential matrix from fundamental matrix and camera matrix
        h0 = eucl2homo(x0(inliers,:))*K_inv'; % x0 in homogeneous coordinates
        h1 = eucl2homo(x1(inliers,:))*K_inv'; % x1 in homogeneous coordinates
        mot = emat2mot(E,h0,h1); % motion from essential matrix
    else
        mot = eye(4);
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
