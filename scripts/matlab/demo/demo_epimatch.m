function [zk,vk,nk] = demo_epimatch(I0,K,I,M,x)
    m = size(M,3);
  
    % select a point
    figure; imshow(I0); hold on;
    if nargin < 5
        x = zeros(2,1);
        [x(1),x(2)] = ginput(1);
    end
    plot(x(1),x(2),'ro');

    % get fundamental matrices
    F = zeros(3,3,m);
    for i = 1 : m
        F(:,:,i) = mot2fmat(M(:,:,i),K,K);
    end

    % find epipolar lines
    x = [x(1);x(2);1];
    l = zeros(3,m);
    for i = 1 : m
        l(:,i) = F(:,:,i) * x;
    end

    % find border points and search range
    w = 5;
    n = 500;
    b = lineToBorderPoints(l',size(I0))';
    x0 = min(b([1,3],1)) + w;
    x1 = max(b([1,3],1)) - w;

    xk = linspace(x0,x1,n);
    zk = pts2depth(x(1),x(2),xk,K,M(:,:,1));

    %z0 = pts2depth(x(1),x(2),x0,K,M(:,:,1));
    %z1 = pts2depth(x(1),x(2),x1,K,M(:,:,1));
    
    %d0 = 1/max(z0,z1);
    %d1 = 1/min(z0,z1);
    %zk = linspace(z0,z1,100);%1./linspace(d0,d1,100);

    % start epipolar search
    figure; im = reshape(permute(I,[1,3,2]),[],size(I0,2));
    imshow(im); hold on;
    for i = 1 : m
        hi = (i-1)*size(I0,1);
        plot(b([1,3],i),b([2,4],i)+hi,'g-');
    end

    I0 = double(I0);
    I  = double(I);
    
    [xx,yy] = meshgrid((x(1)-w):(x(1)+w),(x(2)-w):(x(2)+w));
    v0 = interp2(I0,xx,yy,'cubic');
    vk = zeros(numel(zk),1);
    nk = zeros(numel(zk),1);
    
    for j = 1 : numel(zk)
        gj = zk(j) * inv(K) * [x(1);x(2);1];
        
        for i = 1 : m
            xi = K * M(1:3,:,i) * [gj;1];
            xi = xi / xi(3);
            
            [xx,yy] = meshgrid((xi(1)-w):(xi(1)+w),(xi(2)-w):(xi(2)+w));
            vi = interp2(I(:,:,i),xx,yy,'cubic');

            if ~all(isfinite(vi)), continue; end
            vk(j) = vk(j) + mean(abs(vi(:)-v0(:)));
            nk(j) = nk(j) + 1;
            
            hi = (i-1)*size(I0,1);
            plot(xi(1),xi(2)+hi,'ro');
        end
        
        drawnow;
        %hold off;
    end
    
    [~,idx] = sort(1./zk);
    figure, grid on, hold on, plot(1./zk(idx),vk(idx)./nk(idx),'or-');
end

function z = pts2depth(x0,y0,x1,K,M)
    a = K * M(1:3,1:3) * inv(K) * [x0;y0;1];
    t = K * M(1:3,4);
    z = (x1*t(3) - t(1)) ./ (a(1)-x1*a(3));
end