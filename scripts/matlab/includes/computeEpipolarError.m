function y = computeEpipolarError(x0,x1,F,method)
	if nargin < 4, method = 'sampson'; end

	if size(x0,2) == 2, x0 = eucl2homo(x0); end
	if size(x1,2) == 2, x1 = eucl2homo(x1); end

	assert(size(x0,2) == 3 && size(x1,2) == 3 && size(x0,1) == size(x1,1));

	xFx = dot(x1*F,x0,2);
	
	if strcmpi(method,'algebraic')
		y = xFx;
		return;
	end
	
	sampson = strcmpi(method,'sampson');

	Fx0 = x0 * F'; nx0 = Fx0(:,1).^2 + Fx0(:,2).^2;
	Fx1 = x1 * F ; nx1 = Fx1(:,1).^2 + Fx1(:,2).^2;
	
	if sampson
		y = (xFx.^2)./(nx0+nx1);
	else
		y = [xFx./sqrt(nx1),xFx./sqrt(nx0)];
	end
end