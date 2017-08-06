function demo_bayesian(x0,C0,x1,C1)

    K = C0 * inv(C0 + C1);
    x = x0 + (x1 - x0) * K';
    C = K * C1;

    figure; grid on; hold on;
    drawEllipsoid(x0,C0,[.4,.4,.4],0.1);
    drawEllipsoid(x1,C1,[.4,.4,.4],0.2);
    drawEllipsoid(x,C,'red',0.3);
end

function drawEllipsoid(u,C,color,alpha)
    [V,D] = eig(C);
    radii = sqrt(diag(D));
    [x,y,z] = ellipsoid(0,0,0,radii(1),radii(2),radii(3),10);
    
    vtx = [x(:),y(:),z(:)] * V';
    x = reshape(vtx(:,1),size(x,1),size(x,2)) + u(1);
    y = reshape(vtx(:,2),size(y,1),size(y,2)) + u(2);
    z = reshape(vtx(:,3),size(z,1),size(z,2)) + u(3);
    
    surf(x,y,z,'EdgeColor',color,'LineStyle','-','FaceAlpha',alpha,'FaceColor',color);
end