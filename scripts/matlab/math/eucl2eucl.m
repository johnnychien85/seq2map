function x = eucl2eucl(x,M)
    x = eucl2homo(x) * M(1:3,:)';
end
