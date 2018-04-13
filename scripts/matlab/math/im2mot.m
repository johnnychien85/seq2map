% IM2MOT finds relative pose from two images.
function M = im2mot(I1,I2,K1,K2,epsilon,confidence,iter)
    if nargin < 4, K2 = K1;              end
    if nargin < 5, epsilon    = 0.05;    end
    if nargin < 6, confidence = 0.99;    end
    if nargin < 7, iter       = 1000;    end
    
    x1 = detectFASTFeatures(I1);
    x1 = double(x1.Location);

    klt = vision.PointTracker(...
			'MaxBidirectionalError', epsilon, ...
			'BlockSize',             [5,5]);

    initialize(klt,x1,I1);
    [x2,tracked] = step(klt,I2);
    tracked = find(tracked);

    x1 = x1(tracked,:)-1;
	x2 = x2(tracked,:)-1;

    E = estimateEssentialMatrix(x1,x2,K1,K2,epsilon,confidence,iter);
    M = emat2mot(E,eucl2cano(x1,K1),eucl2cano(x2,K2));
    
    % for visualisation..
    % [E,inliers] = estimateEssentialMatrix(x1,x2,K1,K2,epsilon,confidence,iter);
    % M = emat2mot(E,eucl2cano(x1,K1),eucl2cano(x2,K2));
    % im = zeros(size(I1,1),size(I1,2),3,'uint8');
	% im(:,:,1) = I1;
	% im(:,:,2) = I1/2+I2/2;
	% im(:,:,3) = I1;
    % inliers = find(inliers);
    % figure, imshow(im); hold on;
    % plot(x1(:,1)+1,x1(:,2)+1,'g.');
    % plot([x1(inliers,1),x2(inliers,1)]'+1,[x1(inliers,2),x2(inliers,2)]'+1,'r-');
end