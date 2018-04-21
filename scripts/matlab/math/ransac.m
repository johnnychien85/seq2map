% a generic RANSAC subroutine
function [x_bst,d_bst,inliers] = ransac(m,n,f_est,f_eval,epsilon,confidence,iter,acceptance)
	x_bst = [];
	d_bst = [];
	i_bst = [];
	n_bst = 0;
	inliers = false(m,1);
	k = 0;
	while k < iter
        idx = drawSamples(m,n);
        x_est = f_est(idx);
        d_est = f_eval(x_est);
        i_est = find(d_est < epsilon);
        n_est = numel(i_est);

		if n_est > n_bst && n_est/m > acceptance
			x_bst = x_est;
			d_bst = d_est;
			i_bst = i_est;
			n_bst = n_est;
			iter = min(iter,ceil(log(1-confidence/100)/log(1-(n_bst/m)^n)));
		end

		k = k + 1;
	end
   
    if ~isempty(i_bst)
        x_bst = f_est(i_bst);
        d_bst = f_eval(x_bst);
        i_bst = find(d_bst < epsilon);
        inliers(i_bst) = 1;
    end
end

function idx = drawSamples(m,n)
	assert(m >= n);
	idx = randperm(m);
	idx = idx(1:n)';
end