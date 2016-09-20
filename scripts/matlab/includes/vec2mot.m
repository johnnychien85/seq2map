% restore pose matrix from it's vector representation
function M = vec2mot(x)
    t = x(4:6); % * 100;
    n = norm(x(1:3));
	a = deg2rad(x(1:3)); 
    R = angle2dcm(a(1),a(2),a(3));
	M = [R,t;0,0,0,1];
end