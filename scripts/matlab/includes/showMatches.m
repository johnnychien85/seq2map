function showMatches(Ii,Ij,xi,xj,inliers)
	im = zeros(size(Ii,1),size(Ii,2),3,'uint8');
	im(:,:,1) = Ii;
	im(:,:,2) = Ii/2+Ij/2;
	im(:,:,3) = Ij;
	
	figure, imshow(im); hold on;
	
	outliers = setdiff(1:size(xi,1),inliers);

	plot([xi(outliers,1),xj(outliers,1)]',[xi(outliers,2),xj(outliers,2)]','r-');
	plot([xi(inliers,1), xj(inliers, 1)]',[xi(inliers, 2),xj(inliers, 2)]','g-');
	plot(xi(:,1),xi(:,2),'r.',xj(:,1),xj(:,2),'b.');
end