function drawMotion(m,c,mkr,lbl,a)
	if nargin < 3, mkr = '-'; end;
	if nargin < 4, lbl = 0;   end;
	if nargin < 5, a   = gca; end;

	axis equal; grid on; hold on; view(a,[0,-1,0]);
	tn = size(m,3);
	for k = 1 : tn, m(:,:,k) =  inv(m(:,:,k)); end
	t = reshape(m(1:3,4,:), 3, tn);
	plot3(a,t(1,:),t(2,:),t(3,:),mkr,'color',c,'linewidth',2);
	if lbl > 0
		for k = 1 : lbl : tn, text(t(1,k)+.5,t(2,k)+.5,t(3,k),num2str(k-1)); end
	end
end

