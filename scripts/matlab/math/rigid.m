% y = M * x
function M = rigid(x,y)
    u = mean(x,1);
    v = mean(y,1);
    M = bsxfun(@minus,x,u)' * bsxfun(@minus,y,v);

    [Sxx,Syx,Szx,Sxy,Syy,Szy,Sxz,Syz,Szz] = dealr(M(:));
    N = [Sxx+Syy+Szz,     Syz-Szy,      Szx-Sxz,      Sxy-Syx;...
             Syz-Szy, Sxx-Syy-Szz,      Sxy+Syx,      Szx+Sxz;...
             Szx-Sxz,     Sxy+Syx, -Sxx+Syy-Szz,      Syz+Szy;...
             Sxy-Syx,     Szx+Sxz,      Syz+Szy, -Sxx-Syy+Szz];

    [V,D] = eig(N);
    [~,i] = max(real(diag(D)));
    q = real(V(:,i(1)));

    [~,j] = max(abs(q));
    q = q*sign(q(j(1)));

    R = quat2dcm(q);
    t = R*u' - v';

    M = [R,t];
end

function varargout = dealr(v)
    varargout = num2cell(v);     
end

function R = quat2dcm(q)
    n = norm(q);
    q = q / n;

    assert(n > 0);

    q0 = q(1);
    qx = q(2); 
    qy = q(3); 
    qz = q(4);
    w  = q(2:4);

    Z = [ q0, -qz,  qy;...
          qz,  q0, -qx;...
         -qy,  qx,  q0];

    R = w*w' + Z^2;
end