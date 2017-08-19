function B = fullsymmat(A,m)
    B = zeros(m,m,size(A,1),'like',A);

    for i = 1 : m
        for j = 1 : m
            B(i,j,:) = reshape(A(:,sub2symind(i-1,j-1,m)+1),1,1,[]);
        end
    end
end

function ind = sub2symind(i,j,m)
    if i < j, ind = i*m - (i-1) * i/2 + j - i;
    else      ind = j*m - (j-1) * j/2 + i - j;
    end
end