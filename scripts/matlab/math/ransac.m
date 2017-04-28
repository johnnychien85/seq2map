% a generic RANSAC subroutine
function [x_best,d_best,inliers] = ransac(m,n,f_est,f_eval,epsilon,confidence,iter,acceptance)
	x_best = [];
	d_best = [];
	i_best = [];
	n_best = 0;
	inliers = false(m,1);
	k = 0;
	while k < iter
		% if iter < 1000
			idx = drawSamples(m,n);
			x_est = f_est(idx);
			d_est = f_eval(x_est);
			i_est = find(d_est < epsilon);
			n_est = numel(i_est);
		% else
			% nn = 400;
			% n_est = zeros(nn,1);
			% x_est = cell(nn,1);
			% d_est = cell(nn,1);
			% i_est = cell(nn,1);
			% parfor i = 1 : nn
				% idx = drawSamples(m,n);
				% x_est{i} = f_est(idx);
				% d_est{i} = f_eval(x_est{i});
				% i_est{i} = find(d_est{i} < epsilon);
				% n_est(i) = numel(i_est{i});
			% end
			% [n_est,i] = max(n_est);
			% x_est = x_est{i};
			% d_est = d_est{i};
			% i_est = i_est{i};
		% end

		if n_est > n_best && n_est/m > acceptance
			x_best = x_est;
			d_best = d_est;
			i_best = i_est;
			n_best = n_est;
			iter = min(iter,ceil(log(1-confidence/100)/log(1-(n_best/m)^n)));
		end
		
		k = k + 1;
	end
	inliers(i_best) = 1;
end

function idx = drawSamples(m,n)
	assert(m >= n);
	idx = randperm(m);
	idx = idx(1:n)';
end