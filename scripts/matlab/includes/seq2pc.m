function pc = seq2pc(seq)
    M = seq.cam(1).M;
    cam = seq.cam(1);
    lid = seq.lid(1);
    
    gridStep = 5;
    t0 = 1;
    tn = size(M,3);
    imageTexturing = true;
    
    fprintf('Aligning LiDAR points..');
    for t = t0 : tn
        ld = loadLiDARData(lid,t,'ReshapePoints',true,'ValidPointsOnly',true);
        xyz = eucl2eucl(ld.points,invmot(M(:,:,t)) * lid.E);
        
        if ~imageTexturing
            rgb = repmat(uint8(ld.reflx*255),1,3);
        else
            im = loadImage(cam,t,'greyscale');
            [~,valid,idx] = points2image(xyz,cam,im,'Quiet',true);
            valid = find(valid);
            xyz = xyz(valid,:);
            rgb = repmat(im(idx(valid)),1,3);
        end

        if ~exist('pc'), pc = pointCloud(xyz,'Color',rgb);
        else             pc = pcmerge(pc,pointCloud(xyz,'Color',rgb),gridStep);
        end
        
        if mod(t-t0+1,round((tn-t0+1)/10)) == 0, fprintf('..%d',t); end
    end
    fprintf('..DONE\t');
end