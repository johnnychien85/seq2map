function chk = ismot(x)
    tol = eps(1);
    chk = ndims(x) == 2 && (size(x,1) == 3 || size(x,1) == 4) && size(x,2) == 4 && ...
          abs(norm(x(1:3,1:3)) - 1) <= tol && det(x(1:3,1:3)) > 0;
end