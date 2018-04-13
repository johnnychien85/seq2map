% BISECT2 cuts through a given matrix along columns to achieve
% minimal cost specified by the matrix itself while maintaining
% desired smoothness. The minimisation is based on the Viterbi
% algorithm, a dynamic programming technique.
%
% By Hsiang-Jen (Johnny) Chien (jchien@aut.ac.nz) from Centre of Robotics and
% Vision (CeRV), Auckland University of Technology, New Zealand.
%
function cut = bisect2(x,p,monotonic)
    [m,n] = size(x);

    y = zeros(m,n);
    b = zeros(m,n,'uint16');
    w = abs(repmat(1:m,m,1) - repmat(1:m,m,1)');
    
    if nargin > 2
        switch (monotonic)
            case '+', w(find(triu(w) > 0)) = inf;
            case '-', w(find(tril(w) > 0)) = inf;
            otherwise error 'unknown monotonicity, should be either "+" or "-"';
        end
    end
    
    cut = zeros(1,n,'uint16');

    % initial cost
    y(:,1) = x(:,1);
    b(:,1) = 1:m;   
            
    % forward pass
    for k = 2 : n
        [y(:,k),b(:,k)] = min(repmat(y(:,k-1)'+x(:,k)',m,1)+p*w,[],2);
        %y(:,k) = y(:,k) + x(:,k);
    end   
    
    % backward pass
    for k = n : -1 : 1
        if k == n
            [~,cut(k)] = min(y(:,k));
        else
            cut(k) = b(cut(k+1),k);
        end
    end
end
