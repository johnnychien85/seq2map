function [rgb,ind] = val2rgb(val,map,lim,toUInt8)
    if nargin < 3, lim = [prctile(val,5),prctile(val,95)]; end
    if nargin < 4, toUInt8 = false; end
    ind = round(satnorm(val,lim(1),lim(2)) * (size(map,1) - 1)) + 1;
    rgb = ind2rgb(ind,map);
    
    if toUInt8, rgb = uint8(rgb*255); end
end

function x = satnorm(x,x0,x1)
    x = (x - x0) / (x1 - x0);
    x = min(max(x,0),1);
end