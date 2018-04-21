function [M,T] = elshow(el,xgrid,zgrid,im,cam,tfm,range)
    if nargin < 7, range = [-1,1]; end

    [X,Z] = meshgrid(xgrid,zgrid);
    Y = -el;

    if numel(range) == 2
        Y(find(Y < -range(2) | Y > -range(1))) = NaN;
    else
        assert(all(size(range) == size(el)));
        Y(find(~range)) = NaN;
        range = minmax(Y);
    end

    G = {X,Y,Z};
    G = eucl2eucl(cat(3,G{:}),invmot(tfm));
    C = val2rgb(el,jet,range);

    x = G(:,:,1) ./ G(:,:,3) * cam(1).K(1,1) + cam(1).K(1,3);
    y = G(:,:,2) ./ G(:,:,3) * cam(1).K(2,2) + cam(1).K(2,3);

    x = x(1:2:end,1:2:end,:);
    y = y(1:2:end,1:2:end,:);
    Y = Y(1:2:end,1:2:end,:);
    C = C(1:2:end,1:2:end,:);
    
    imshow(im); hold on;
    mesh(x,y,ones(size(x)),Y,'EdgeAlpha',0.75,'LineWidth',1,'FaceColor','none','CData',C);
    hold off;

    if nargout > 0
        idx = find(~isnan(x) & ~isnan(y));
        % T.tri = delaunay(x(idx),y(idx));
        % T.x = x(idx);
        % T.y = y(idx);

        % M = zeros(size(im,1),size(im,2),'uint8');
        % M = insertShape(M,'FilledPolygon',[T.x(T.tri(:,1)),T.y(T.tri(:,1)),T.x(T.tri(:,2)),T.y(T.tri(:,2)),T.x(T.tri(:,3)),T.y(T.tri(:,3))],'Color','red');

        T = alphaShape(x(idx),y(idx),5);
        M = zeros(size(im,1),size(im,2),'uint8');
        F = T.boundaryFacets;
        M = insertShape(M,'FilledPolygon',reshape(T.Points(F(:,1),:)',1,[]),'Color','red');
    end
end