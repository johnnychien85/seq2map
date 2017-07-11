function M = txt2mot(filename)
	f = fopen(filename, 'r');
    x = fscanf(f,'%f',[12,inf])';
    M = zeros(4, 4, size(x,1));
    M(4,4,:) = 1;
        
    for i = 1 : size(x,1)
        M(1:3,:,i) = reshape(x(i,:),[4,3])';
    end
end