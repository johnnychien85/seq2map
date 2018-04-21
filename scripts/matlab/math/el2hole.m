% EL2HOLE finds potholes from a DEM.
function holes = el2hole(el,zgrid,xgrid,im,cam,rcs)
    % elevation threshold to consider a local peak to be a seed to
    % find a possible pothole
    Emax = -.025; % focus on cells below -2.5cm of the road surface

    % tolerance to consider a cell connected to a pothole cell
    Etol = 0.010;

    % range of size for pothole validity check
    NLim = [50*50,100*100]; % from 50cm2 to 100cm2

    % find invalid cells
    L = bwlabel(~isfinite(el));

    % mask for cells outside the road
    B = L == L(1,1) | L == L(1,end);
    B = imdilate(B,ones(1,11));

    % filtered elevation
    el(find(~isfinite(el))) = NaN;
    E = fillmissing(el,'linear');
    E(find(B)) = -Inf;

    % find local minimum
    P = imregionalmin(medfilt2(E)) & ~B & E < Emax;
    [y,x] = find(P);

    % pothole labels
    H = zeros(size(L),'uint16');

    % searching for potholes from the seeds specified by (x,y)
    n = 0;          % number of found potholes
    N = zeros(0,1); % number of cells for each pothole
    for i = 1 : numel(y)
        % a visited pothole?
        if H(y(i),x(i)) > 0, continue; end

        % a new pothole
        n = n + 1;
        A = grayconnected(E,y(i),x(i),Etol); % adjacency query
        C = setdiff(unique(H(find(A))),0);   % connected potholes which were discovered earlier
        H(find(A)) = n; % fill in the label for the pothole

        % merge previously discovered potholes that are connected to the new one
        for j = 1 : numel(C)
            H(find(H == C(j))) = n;
            N(C(j)) = 0;
        end
        N(n) = numel(find(H == n));
    end

    idx = find(N > NLim(1) & N < NLim(2));

    if isempty(idx)
        holes = [];
        return;
    end

    n = 0;
    if size(im,3) == 1, im = repmat(im,1,1,3); end
    [r,g,b] = slicearray(im);
    
    for k = 1 : numel(idx)
        P = H == idx(k);
        [i,j] = find(P);
        zspan = range(zgrid(i));
        xspan = range(xgrid(j));

        if ~isempty(find(P & B)) || zspan > 1.25, continue; end

        n = n + 1;
        M = elshow(el,xgrid,zgrid,im,cam,rcs,P);
        M = M(:,:,1) > 0;

        close(gcf);

        holes(n).range = zspan;
        holes(n).width = xspan;
        holes(n).depth = -min(E(sub2ind(size(E),i,j)));
        holes(n).cells = numel(i);
        holes(n).mask  = P;
        holes(n).roi   = M;

        idx = find(edge(M) == 1);
        r(idx) = 255;
        g(idx) = 0;
        b(idx) = 0;
    end

    im = {r,g,b};
    im = cat(3,im{:});
    imshow(im);
end