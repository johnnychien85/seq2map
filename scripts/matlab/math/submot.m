% SUBMOT computes local transformation between two
% frames given a global transformation stack M.
function Mij = submot(M,ti,tj)
    % subtract the motion of the subsequence defined by ti
    if nargin < 3
        Mij = zeros(4,4,numel(ti));
        M0  = invmot(M(:,:,ti(1)));
        for k = 1 : numel(ti)
            Mij(:,:,k) =  M(:,:,ti(k)) * M0;
        end
        return;
    end

    if ti == tj, Mij = eye(4);
    else         Mij = M(:,:,tj) * invmot(M(:,:,ti));
    end
end