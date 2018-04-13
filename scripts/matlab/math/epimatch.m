% EPIMATCH performs exhaustive search for each
% image point in pts2d following the corresponding
% epipolar lines in I2 to find the best match.
function D = epimatch(I1,I2,M12,K1,K2,pts2d)
    [F,~,e2] = mot2fmat(M12,K1,K2); % find the fundamental matrix between I1 and I2 and the epipole in I2
    x1 = pts2d + 1;
    p2 = findEndPoints(pts2d,F,size(I2));
    x2 = blkproc([p2,x1],[1,6],@f);

    function y = f(x)
        [u,v,e12] = blkmatch(I1,I2,x(5:6),x(1:2),x(3:4),2,2);
        
        %e12 = conv((x(5) - v2).^2,ones(1,11),'same');
        %figure, plot(e12); hold on;
        [~,idx] = min(e12);
        y = [u(idx),v(idx)];
        %plot(idx,e12(idx),'ro');
    end
    
    figure, imshow(I1); hold on;
    cmap = lines(size(x1,1));
    for i = 1 : size(cmap,1)
        plot(x1(i,1),x1(i,2),'o','color',cmap(i,:),'markersize',10,'linewidth',3);
    end
    hold off;
    
    figure, imshow(I2); hold on;
    line(p2(:,[1,3])',p2(:,[2,4])');
    plot(e2(1),e2(2),'wx','markersize',5);
    for i = 1 : size(cmap,1)
        plot(x2(i,1),x2(i,2),'o','color',cmap(i,:),'markersize',10,'linewidth',3);
    end
    hold off;

    D = zeros(size(pts2d,1),1);
end

function x2 = findEndPoints(x1,F,imageSize)
    l2 = epipolarLine(F,x1); % find epipolar lines in I2
    x2 = lineToBorderPoints(l2,imageSize);
end

function [u,v,e] = blkmatch(I1,I2,x1,x2i,x2j,m,n)
    % base image
    a1 = double(I1(x1(2)-m:x1(2)+m,x1(1)-n:x1(1)+n));

    % gather pixels from reference image along line x2i->x2j
    [u,v,a2] = improfile(I2,[x2i(1),x2j(1)],[x2i(2),x2j(2)]);
    a2 = repmat(reshape(a2,1,1,[]),m,n);
    for dv = -m : m
        for du = -n : n
            if du == 0 && dv == 0, continue; end
            a2_uv = improfile(I2,[x2i(1),x2j(1)]+du,[x2i(2),x2j(2)]+dv);
            a2(dv+m+1,du+n+1,:) = reshape(a2_uv,1,1,[]);
        end
    end

    a1 = reshape(a1,1,[]);
    a2 = reshape(a2,size(a1,2),[],1)';
    e  = sum(abs(bsxfun(@minus,a2,a1)),2);
%{
    v2_11 = improfile(I2,x([1,3])-1,x([2,4])-1);
    v2_12 = improfile(I2,x([1,3])  ,x([2,4])-1);
    v2_13 = improfile(I2,x([1,3])+1,x([2,4])-1);
    v2_21 = improfile(I2,x([1,3])-1,x([2,4])  );
    v2_23 = improfile(I2,x([1,3])+1,x([2,4])  );
    v2_31 = improfile(I2,x([1,3])-1,x([2,4])+1);
    v2_32 = improfile(I2,x([1,3])  ,x([2,4])+1);
    v2_33 = improfile(I2,x([1,3])+1,x([2,4])+1);
    v2 = [v2_11,v2_21,v2_31,v2_12,v2_22,v2_32,v2_13,v2_23,v2_33];
    v1 = double(reshape(I1(x(6)-1:x(6)+1,x(5)-1:x(5)+1),1,9));
    e12 = sum(bsxfun(@minus,v2,v1).^2,2);
%}
end