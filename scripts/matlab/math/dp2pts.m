function [pts,jac] = dp2pts(dp,pri,sec)
    [m,n] = size(dp);
	[u,v] = meshgrid(1:n,1:m);

    dp(find(dp<0)) = 0;
    
    if isfield(pri,'fu'), rect = pri;
    else rect = cam2rect(pri,sec);
    end

    pts = zeros(m,n,3);
    pts(:,:,3) = -rect.b*rect.fu./dp;
    pts(:,:,1) = pts(:,:,3) .* (u-1-rect.uc)/rect.fu;
    pts(:,:,2) = pts(:,:,3) .* (v-1-rect.vc)/rect.fv;
    
    if nargout < 2, return; end
    
    jac = zeros(m,n,9);
    jac(:,:,1) = pts(:,:,3) / rect.fu;
    jac(:,:,5) = pts(:,:,3) / rect.fu;
    jac(:,:,7) = pts(:,:,3) .* (u-1-rect.uc)/rect.fu ./dp;
    jac(:,:,8) = pts(:,:,3) .* (v-1-rect.uc)/rect.fu ./dp;
    jac(:,:,9) = pts(:,:,3) ./dp;
    
    %{
    jac = zeros(m,n,6);
    dp2 = dp .* dp;
    jxy = pts(:,:,3) / rect.fu;
    jdp = -dp2pts(-dp,rect);
    jac(:,:,1) = jdp(:,:,1) .* jdp(:,:,1) + jxy .* jxy;
    jac(:,:,2) = jdp(:,:,1) .* jdp(:,:,2);
    jac(:,:,3) = jdp(:,:,1) .* jdp(:,:,3);
    jac(:,:,4) = jdp(:,:,2) .* jdp(:,:,2) + jxy .* jxy;
    jac(:,:,5) = jdp(:,:,2) .* jdp(:,:,3);
    jac(:,:,6) = jdp(:,:,3) .* jdp(:,:,3);
    %}
end
