function im = stxshow(im,stx,dmax)
    if nargin < 3, dmax = 64; end;
    if size(im,3) == 1, im = repmat(im,1,1,3); end

    filled = true;

    if isempty(stx)
        imshow(im);
        return;
    end

    dp = cat(1,stx.d);
    cmap = jet(256);
    cidx = round(max(0,min(1,dp/dmax))*(size(cmap,1)-1)) + 1;

    x = cat(1,stx.x);
    y = cat(1,stx.y);
    h = cat(1,stx.height);
    w = cat(1,stx.width);

    cmap = uint8(cmap(cidx,:)*255);

    if filled
        im = insertShape(im,'FilledRectangle',[x,y,w,h],'Color',cmap,'Opacity',0.5);
        im = insertShape(im,'Rectangle',      [x,y,w,h],'Color','white','LineWidth',1,'Opacity',0.2);
    else
        im = insertShape(im,'Rectangle',      [x,y,w,h],'Color',cmap,'LineWidth',1,'Opacity',0.2);
    end

    imshow(im);
end