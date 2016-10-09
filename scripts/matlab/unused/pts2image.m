function [x2d, inRange, idx, im] = pts2image(x3d, cam, im, varargin)
    po = parseArgs(varargin{:});

    if nargin < 3
        lim = [-inf,-inf,-inf;inf,inf,inf];
        im  = [];
        po.Quiet = true;
    elseif ndims(im) == 2 && all(size(im) == [2,3])
        lim = im;
        im  = [];
    else
        lim  = [0,0,0;size(im,2)-1,size(im,1)-1,inf];
    end

    if ~isfield(cam,'D') || all(cam.D == 0) % linear projection
        x2d_h = eucl2homo(x3d(:,1:3)) * cam.P';
        x2d   = homo2eucl(x2d_h);    
    else % with nonlinear correction
        x2d_h = eucl2homo(x3d(:,1:3)) * cam.E(1:3,:)';
        x2d_d = distort(homo2eucl(x2d_h), cam.D);
        x2d   = homo2eucl(eucl2homo(x2d_d) * cam.K');
    end

    inRange = x2d  (:,1) > lim(1,1) & x2d  (:,1) < lim(2,1) &...
              x2d  (:,2) > lim(1,2) & x2d  (:,2) < lim(2,2) &...
              x2d_h(:,3) > lim(1,3) & x2d_h(:,3) < lim(2,3);

    if isempty(im)
        if ~po.Quiet,   warning 'pass me image data if you like to see something';        end
        if nargout > 2, warning 'cannot determine pixel indices without image dimension'; end
        return;
    end

	if ~po.Quiet && isempty(po.Figure), po.Figure = figure; end
	
    if ~po.Quiet || nargout > 3
        idx = find(inRange);

        if isempty(po.Values), v = x2d_h(idx,3);
        else                   v = po.Values(idx); end

        if isempty(po.Colormap), cmap = hsv(64);
        else                     cmap = po.Colormap; end

        im = drawPoints(im, x2d(idx,:)+1, v, cmap, po.Figure);
    end

    idx = zeros(size(x3d,1), 1, 'uint32');

    if nargout > 2
        u = round(x2d(:,1));
        v = round(x2d(:,2));

        I  = find(u >= 0 & u < size(im,2) & v >= 0 & v < size(im,1));
        sz = size(im);
        idx(I) = sub2ind(sz(1:2), v(I)+1, u(I)+1);
    end
end

function x2d = distort(x2d, D)
    k1 = D(1); k2 = D(2); p1 = D(3); p2 = D(4); k3 = D(5);
    r  = sqrt(sum(x2d.^2,2));
    r2 = r.^2; r4 = r2.^2; r6 = r2.^3;
    a = 1 + k1 * r2 + k2 * r4 + k3 * r6;
    x = x2d(:,1); y = x2d(:,2); xy = x2d(:,1) .* x2d(:,2);
    u = x .* a + 2 * p1 * xy + p2 * (r2 + 2 * x.^2);
    v = y .* a + 2 * p2 * xy + p1 * (r2 + 2 * y.^2);
    x2d = [u,v];
end

function po = parseArgs(varargin)
    po = inputParser;
    addParameter(po, 'Quiet',  false);
    addParameter(po, 'Figure', []);
    addParameter(po, 'Values', []);
    addParameter(po, 'Colormap', []);
    parse(po, varargin{:});
    po = po.Results;
end

function im = drawPoints(im, x2d, v, cmap, fig)
    v = double(v);
    vmax = max(v);
    vmin = min(v);
    
    if vmax == vmin, vmax = vmin + 1; end
    cidx = round((v - vmin) ./ (vmax - vmin) * (size(cmap,1) - 1)) + 1;

	if size(im,3) == 1, im = histeq(im); end
	im = insertShape(im,...
        'FilledCircle', [x2d,repmat(3,size(x2d,1),1)],...
        'Color',        uint8(255*cmap(cidx,:)),...
        'Opacity',      0.3);

	if isempty(fig), return; end

    if isa(fig, 'vision.VideoPlayer')
		step(fig,im);
    else
        figure(fig);
        imshow(im,'Border','tight');
		%hold on; scatter(x2d(:,1), x2d(:,2), 15, cmap(cidx,:),'filled'); hold off;
    end
end

