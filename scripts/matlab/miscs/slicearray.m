% [A1,A2,A3,...] = SLICEARRAY(A) slices array A into submatrices A1, A2, A3,...
% along the last dimension of A.
function varargout = slicearray(A)
    dims = size(A);
    A = reshape(A,[],dims(end));

    for k = 1 : size(A,2)
        varargout{k} = reshape(A(:,k),[dims(1:end-1),1]);
    end
end