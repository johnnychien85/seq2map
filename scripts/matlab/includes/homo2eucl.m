function x = homo2eucl(x)
    x = x(:,1:end-1) ./ repmat(x(:,end), 1, size(x,2) - 1);
end

