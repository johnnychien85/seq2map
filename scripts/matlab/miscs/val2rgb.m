function [rgb,cidx] = val2rgb(val,cmap,lim)
    if nargin < 3, lim = [prctile(val,5),prctile(val,95)]; end
    cidx = round(max(0,min(1,(val-lim(1))/(lim(2)-lim(1)))) * (size(cmap,1) - 1)) + 1;
    rgb = cmap(cidx,:);
end