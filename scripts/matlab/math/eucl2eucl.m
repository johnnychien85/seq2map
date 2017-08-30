function x = eucl2eucl(x,M)
    if isempty(x)
        x = zeros(0,3);
        return;
    end
    if size(x,3) > 1
        assert(size(x,3) == 3);
        x = reshape(eucl2eucl(reshape(x,[],3),M),size(x,1),size(x,2),3);
        return;
    end
    x = eucl2homo(x) * M(1:3,:)';
end
