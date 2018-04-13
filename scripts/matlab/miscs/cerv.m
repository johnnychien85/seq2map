% cerv  Varient of HSV
% cerv(M), a variant of HSV(M), is an M-bt-3 matrix contraining
% colours start with red (hue=10) and end with dark blue (hue=220).
% An extra pink (hue=285) is inserted in the end to distinguish
% values reaching the maximum of the range.
function rgb = cerv(M)
    if nargin < 1, M = 256; end
    hue = [linspace(10,220,M-1),285] / 360;
    rgb = hsv2rgb([hue',ones(M,2)]);
end