function mot = epnp(x,y,K)
    % find four control points
    B = [rand(4,3),ones(4,1)];

    % barycentric coordinates
    x = eucl2homo(x)*inv(B);
    
    % normalised image coordinates
    y = eucl2homo(y)*inv(K)';

    % solve for the transformed control points using DLT
    C = dlt(x',y')';

    % scale recovery
    k = mean(blkproc(diff(B),[1,4],'norm') ./ blkproc(diff(C),[1,4],'norm'));
    C = k * C;

    if sign(C(1,1)) ~= sign(B(1,1)), C = -C; end

    % motion recovery by solving rigid body alignment
    mot = rigid(B(:,1:3),C);
end
