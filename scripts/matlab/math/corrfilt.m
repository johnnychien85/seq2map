function C = corrfilt(A,B,NHOOD)
    assert(all(size(A) == size(B)));

    if nargin < 3, NHOOD = ones(3); end

    NHOOD = NHOOD > 0;

    [m,n] = size(A);
    [h,w] = size(NHOOD);

    assert(mod(h,2) == 1 & mod(w,2) == 1);

    % find local standard deviations before padding..
    Sa = stdfilt(A,NHOOD);
    Sb = stdfilt(B,NHOOD);

    % padding A and B
    bh = (h-1)/2;
    bw = (w-1)/2;

    A = padarray(A,[bh,bw],'symmetric');
    B = padarray(B,[bh,bw],'symmetric');

    % find local means
    Ma = conv2(A,NHOOD,'valid')/nnz(NHOOD);
    Mb = conv2(B,NHOOD,'valid')/nnz(NHOOD);

    % initialise correlation scores
    C = zeros(m,n,'like',A);

    % start accumulation
    for i = 1 : h
        rsrc = (1:m)+i-1;
        for j = 1 : w
            if ~NHOOD(i,j), continue; end
            csrc = (1:n)+j-1;
            C = C + ...
                (A(rsrc,csrc) - Ma) .* ...
                (B(rsrc,csrc) - Mb);
        end
    end
    
    C = C ./ ((Sa.*Sb) * nnz(NHOOD));
    C(find(~isfinite(C))) = NaN;
end