function E = mot2emat(M)
    R = M(1:3,1:3);
    t = normalise(M(1:3,4));
    E = skewsymm(t) * R;

	% vector normalisation
	function x = normalise(x)
		x = x / norm(x);
	end

	% skew-symmetric matrix form of a vector
    function x = skewsymm(x)
        x = [0,-x(3),x(2);x(3),0,-x(1);-x(2),x(1),0];
    end
end