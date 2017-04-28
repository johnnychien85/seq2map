function x = homo2eucl(x)
    x = bsxfun(@rdivide,x(:,1:end-1),x(:,end));
end

