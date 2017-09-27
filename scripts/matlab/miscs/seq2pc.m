function pc = seq2pc(seq,src,mot,t0,cm)
    if nargin < 3, mot = seq.cam(1).M; end
    if nargin < 4, t0  = 1;            end
    if nargin < 5, cm  = [1,1,1];      end

    zfar = 20;
    gridStep = 0.05;
    tn = t0 + size(mot,3) - 1;
    texturing = true;

    if     isfield(src,'Format'), lid = src;
    elseif isfield(src,'pri'),    dpm = src;
    else   error 'unknown source of scene structure'; end

    if exist('lid')
        cam = seq.cam(1);
        fprintf('Aligning LiDAR 3D points..');
        for t = t0 : tn
            ld  = loadLiDARData(lid,t,     ...
                    'ReshapePoints',true,  ...
                    'ValidPointsOnly',true,...
                    'ApplyTransform',true);
            xyz = eucl2eucl(ld.points,invmot(mot(:,:,t-t0+1)));
            
            if ~texturing
                rgb = uint8(ld.reflx * 255 * cm);
            else
                im = histeq(loadImage(cam,t,'greyscale'));
                [~,valid,idx] = pts2image(xyz,cam,im,'Quiet',true);
                valid = find(valid);
                xyz = xyz(valid,:);
                rgb = uint8(double(im(idx(valid))) * cm);
            end

            if ~exist('pc'), pc = pointCloud(xyz,'Color',rgb);
            else             pc = pcmerge(pc,pointCloud(xyz,'Color',rgb),gridStep);
            end
            
            if mod(t-t0+1,round((tn-t0+1)/10)) == 0, fprintf('..%d',t); end
        end
        fprintf('..DONE\n');
    else
        fprintf('Aligning disparity-derived 3D points..');
        for t = t0 : tn
            im = histeq(loadImage(seq.cam(dpm.pri),t,'greyscale'));
            dp = loadDisparity(dpm,t);
            pts = dp2pts(dp,seq.cam(dpm.pri),seq.cam(dpm.sec));
            x = pts(:,:,1);
            y = pts(:,:,2);
            z = pts(:,:,3);
            idx = find(z > 0 & z < zfar);
            rgb = uint8(double(im(idx)) * cm);
            xyz = eucl2eucl([x(idx),y(idx),z(idx)],invmot(mot(:,:,t-t0+1)));

            if ~exist('pc'), pc = pointCloud(xyz,'Color',rgb);
            else             pc = pcmerge(pc,pointCloud(xyz,'Color',rgb),gridStep);
            end
            
            if mod(t-t0+1,round((tn-t0+1)/10)) == 0, fprintf('..%d',t); end
        end
        fprintf('..DONE\n');
    end
end