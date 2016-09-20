% convert post matrix to a 6-vector
function x = mot2vec(M)
    R = M(1:3,1:3);
	[a1,a2,a3] = dcm2angle(R);
    t = M(1:3,4);
	x = [rad2deg([a1;a2;a3]);t];
end