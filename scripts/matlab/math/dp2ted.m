function [ted,d02t] = dp2ted(d01,d12,d02)
	if isempty(d12) || isempty(d12)
		ted = zeros(size(d01));
		d02t = ted;
		return
	end

	assert(isequal(size(d01),size(d12)) && isequal(size(d02),size(d12)));

	d02t = d01 + imwarpdp(d12,d01);
	ted = 1 ./ (1+abs(d02 - d02t));
	bad = find(d01 <= 0 | d12 <= 0 | d02 <= 0);

	ted(bad)  = 0;
	d02t(bad) = 0;
end
