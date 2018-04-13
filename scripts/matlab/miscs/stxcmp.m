% [TP,TN,FP,FN,ERR] = stxcmp(STX,GND,SZ) compares extracted stixels STX against
% ground truth stixels GND. The comparison concludes the differences as
% true positives TP, true negatives TN, false positives FP, and false negatives
% FN. The comparison based on the disparity maps in size SZ converted from both
% stixel representations. The error of disparities ERR is also calculated.
%
% See also: stx2dp.
%
function [tp,tn,fp,fn,err] = stxcmp(stx,gnd,sz)
    if ~iscell(stx), stx = {stx}; end
    if ~iscell(gnd), gnd = {gnd}; end

    frames = numel(stx);
    assert(frames == numel(gnd));

    tp = zeros(frames,1);
    tn = zeros(frames,1);
    fp = zeros(frames,1);
    fn = zeros(frames,1);
    err = cell(frames,1);

    for i = 1 : frames
        dp = stx2dp(stx{i},sz);
        gt = stx2dp(gnd{i},sz);

        P = dp > 0; % positive responses
        N = ~P;     % negative responses
        T = gt > 0; % ground positives
        F = ~T;     % ground negatives

        tp(i) = nnz(P & T);
        tn(i) = nnz(N & F);
        fp(i) = nnz(P & F);
        fn(i) = nnz(N & T);

        hit = find(P & T);
        err{i} = reshape(dp(hit)-gt(hit),[],1);
    end
    % err = cat(1,err{:});
end