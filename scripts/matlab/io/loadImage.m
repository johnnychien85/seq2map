function im = loadImage(cam, frame, varargin)
    im = imread(cam.ImageFiles{frame});
    if isfield(cam, 'Demosaic') && ~isempty(cam.Demosaic), im = demosaic(im, cam.Demosaic); end
    for i = 1 : numel(varargin)
        if ~isstr(varargin{i}), continue; end
        switch lower(varargin{i})
            case {'greyscale','grayscale'}
                if size(im,3) > 1, im = rgb2gray(im); end
            case 'uint8'
                if isa(im,'uint16'), im = uint8(im / 256); end
        end
    end
    
    %im = imresize(im,.5);
end
