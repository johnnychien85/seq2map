% restore pose matrix from it's vector representation
function M = vec2mot(x,angleAxis)
    if nargin < 2, angleAxis = false; end
    t = x(4:6); % * 100;
    if angleAxis
        n = norm(x(1:3));
        r = [x(1:3)/n;n];
        R = vrrotvec2mat(r);
    else
        r = deg2rad(x(1:3)); 
        R = angle2dcm(r(1),r(2),r(3));
    end
	M = [R,t;0,0,0,1];
end