function im = imblend(varargin)
    n = numel(varargin);
    a = mkmap(n);
    
    for k = 1 : n
        Ik = varargin{k};
        
        if size(Ik,3) > 1, Ik = rgb2gray(Ik); end
        if k == 1, im = zeros(size(Ik,1),size(Ik,2),3,'uint8'); end

        Ik = histeq(Ik);
        
        im(:,:,1) = im(:,:,1) + a(k,1) * Ik;
        im(:,:,2) = im(:,:,2) + a(k,2) * Ik;
        im(:,:,3) = im(:,:,3) + a(k,3) * Ik;
    end
end

function map = mkmap(n)
    m = n - 1;
    x = [0:m]'/m*pi;
    map = zeros(n,3);
    map(:,1) = cos(x)*.5+.5;
    map(:,2) = sin(x)*.5+.5;
    map(:,3) = flipdim(map(:,1),1);
    map = bsxfun(@rdivide,map,sum(map));
end
