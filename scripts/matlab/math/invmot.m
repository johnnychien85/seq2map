function M_inv = invmot(M)
    R = M(1:3,1:3);
    t = M(1:3,4);
    M_inv = [R',-R'*t;0,0,0,1];
end