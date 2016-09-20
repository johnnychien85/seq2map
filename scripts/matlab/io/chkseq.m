function chkseq(seq,varargin)
    po = parseArgs(seq,varargin{:});
    cam = seq.cam(po.Camera);

    if po.LiDAR > 0, lid = seq.lid(po.LiDAR);
    else             lid = []; end

	if isempty(lid), po.ShowLiDARPoints = false; end

	hasVision = exist('vision.VideoPlayer') ~= 0;
	hasMotion = size(cam.M,3) > 0;

    if hasVision, fig = vision.VideoPlayer('Position', [0,0,flipdim(cam.ImageSize,2)]);
    else          fig = figure; end

    if ~isempty(po.Output) && hasVision
        writer = vision.VideoFileWriter(po.Output,'FrameRate',10);
    else
        writer = [];
    end

    for i = 1 : numel(cam.ImageFiles)
        im = loadImage(cam,i,'greyscale','uint8');
%im = imresize(im,.5);
        if ~isempty(lid)
            ld = loadLiDARData(lid,i,'ValidPointsOnly',true);
            x3d = eucl2eucl(ld.points,lid.E);
        end

		if po.ShowLiDARPoints
			[~,~,~,im] = points2image(x3d, cam, im, 'Figure', fig);
		else
			[~,~,~,im] = points2image(zeros(0,3), cam, im, 'Figure', fig);
		end
		
        if ishandle(fig), drawnow;
        elseif ~fig.isOpen, return;
        end

        if po.Fancy
            if ~exist('axx'), whitebg(figure); axx = gca; grid on; hold on; end
            idx = find(abs(x3d(:,1)) < 1000 & abs(x3d(:,3)) < 1000);
            idx = idx(1:5:end);
            x3d = x3d(idx,:);
            cmap = hot(64);
            %cval = round(max(0,min(1,(sqrt(x3d(:,1).^2+x3d(:,3).^2)) / 1000)) * (size(cmap,1)-1)) + 1;
			cval = round(max(0,min(1,((x3d(:,2)+200) / 400))) * (size(cmap,1)-1)) + 1;
			
			if hasMotion
				Mi  = inv(seq.cam(1).M(:,:,i));
				x3d = eucl2eucl(x3d,Mi);
			else
				hold off;
			end

            scatter3(axx,x3d(:,1),x3d(:,2),x3d(:,3),1,cmap(cval,:));
			axis equal;
			xlim(axx,[-1000,1000]);
			zlim(axx,[-1000,1000]);

            if hasMotion && i > 1
                cp0 = -cam.M(1:3,1:3,i-1)' * cam.M(1:3,4,i-1);
                cp1 = -cam.M(1:3,1:3,i)'   * cam.M(1:3,4,i);
                plot3(axx,[cp0(1),cp1(1)],[cp0(2),cp1(2)],[cp0(3),cp1(3)],'w-','linewidth',5);
            end
            view([0,-1,0]);
            minimap = frame2im(getframe);
            
            im(1:size(minimap,1),1:size(minimap,2),:) = minimap;
        end

        if ~isempty(writer), step(writer,imresize(im,.5)); end
    end
end

function po = parseArgs(seq,varargin)
    camIdx0 = 1;
    if numel(seq.lid) > 0, lidIdx0 = 1;
    else                   lidIdx0 = 0; end

    po = inputParser;
    addParameter(po, 'Output', []);
    addParameter(po, 'Camera', camIdx0);
    addParameter(po, 'LiDAR',  lidIdx0);
    addParameter(po, 'Fancy',  false);
	addParameter(po, 'ShowLiDARPoints',  true);
    parse(po,varargin{:});
    po = po.Results;

    assert(po.Camera <= numel(seq.cam));
    assert(po.LiDAR  <= numel(seq.lid));
end

