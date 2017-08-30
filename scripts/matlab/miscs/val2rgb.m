function [rgb,ind] = val2rgb(val,map,lim)
    if nargin < 3, lim = [prctile(val,5),prctile(val,95)]; end
    ind = round(satnorm(val,lim(1),lim(2)) * (size(map,1) - 1)) + 1;
    rgb = ind2rgb(ind,map);
end

function x = satnorm(x,x0,x1)
    x = (x - x0) / (x1 - x0);
    x = min(max(x,0),1);
end