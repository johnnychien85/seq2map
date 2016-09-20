function dst = points2edges(x3d,useConvexity)
    if nargin < 2, useConvexity = true; end

%    kdt = KDTreeSearcher(x3d);
%    nn2 = knnsearch(kdt,x3d,'K',3); % search for 2 nearest neighbors for each point
%    nn2 = nn2(:,2:3);
    [T,P,R] = cart2sph(x3d(:,1),x3d(:,2),x3d(:,3));
    kdt = KDTreeSearcher([T,P]);
    nn2 = knnsearch(kdt,[T,P],'K',3); % search for 2 nearest neighbors for each point
    nn2 = nn2(:,2:3);
    
    dR0 = R(nn2(:,1)) - R;
    dR1 = R(nn2(:,2)) - R;
    
    if useConvexity
        dst = max(max(dR0,dR1),0);
    else
        dst = max(abs(dR0),abs(dR1));
    end

    %{
    %[dx0,nx0] = normr(x3d(nn2(:,1),:) - x3d);
    %[dx1,nx1] = normr(x3d(nn2(:,2),:) - x3d);
    %ang = acos(dot(dx0,dx1,2));
    %idx = find(ang > 0);
    %}

    %{
    idx = find(dst > 50);
    figure; hold on; axis equal; grid on;
    plot3([x3d(idx,1),x3d(nn2(idx,1),1)]',[x3d(idx,2),x3d(nn2(idx,1),2)]',[x3d(idx,3),x3d(nn2(idx,1),3)]','xr-');
    plot3([x3d(idx,1),x3d(nn2(idx,2),1)]',[x3d(idx,2),x3d(nn2(idx,2),2)]',[x3d(idx,3),x3d(nn2(idx,2),3)]','.b-');
    plot3(x3d(idx,1),x3d(idx,2),x3d(idx,3),'ko');
    %}
end
%{
function [x,n] = normr(x)
    n = sqrt(sum(x.^2,2));
    x = x ./ repmat(n,1,size(x,2));
end
%}

