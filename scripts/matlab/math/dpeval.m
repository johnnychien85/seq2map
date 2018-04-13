function cst = dpeval(dp,I0,I1,bw)
    assert(all(size(I0) == size(dp)));
    assert(all(size(I0) == size(I1)));
    assert(bw > 0);
    
    I0 = double(I0);
    I1 = double(I1);
    
    I0r = imwarpdp(I1,dp);
    cst = corrfilt(I0,I0r,ones(bw));
end
