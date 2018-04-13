% DP = stx2dp(STX,SZ) converts stixels STX to DP the dense disparity-based
% representation. The size of the disparity map is defined by SZ.
%
% See also: dp2stx.
%
function [dp,lbl] = stx2dp(stx,sz)
    dp  = zeros(sz);
    lbl = zeros(sz,'uint16');
    
    for i = 1 : numel(stx)
        x0 = max(round(stx(i).x),1);
        x1 = min(round(stx(i).x + stx(i).width - 1), sz(2));
        y0 = max(round(stx(i).y),1);
        y1 = min(round(stx(i).y + stx(i).height - 1),sz(1));
        dp (y0:y1,x0:x1) = stx(i).d;
        lbl(y0:y1,x0:x1) = i;
    end
end