% INVCAM finds the inverse of an upper triangular 3-by-3 camera matrix
% K = [a,b,c; 0,d,e; 0,0,1]. The computation is slightly faster than using inv(K).
function K_inv = invcam(K)
    K_inv = [                                                                         ...
        1/K(1,1), -K(1,2)/K(1,1)/K(2,2), K(2,3)*K(1,2)/K(1,1)/K(2,2) - K(1,3)/K(1,1); ...
               0,              1/K(2,2),                             - K(2,3)/K(2,2); ...
               0,                     0,                                           1; ...
    ];
end