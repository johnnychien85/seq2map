function [M,epi,rpe] = solveEgomo(egomo,alpha,M,im,dryrun)
	if nargin < 3, M = [];  end
	if nargin < 4, im = []; end
	if nargin < 5, dryrun = false; end

	extended = true;
	
	n_epi = numel(egomo.epi.uid);
	n_pnp = numel(egomo.pnp.uid);
	
	a_epi = 1 / (1 + n_epi/n_pnp*(1/alpha-1)); %* (n_epi + n_pnp) / n_epi / 2;
	a_rpe = 1 - a_epi;

	if isempty(M)
		% initialisation using essential matrix decomposition
		M = pts2mot(egomo.epi.src,egomo.epi.dst,egomo.K);
	end
	
	x = mot2vec(M);
	
	if ~isempty(egomo.pnp.uid) && ~dryrun
		rpe = f_rpe(x);
		fprintf('initial error:	%.02f	(pnp=%.02f,	epi=%.02f)\n', rms(f(x)), rms(f_rpe(x)), rms(f_epi(x)));
		optim = optimoptions('lsqnonlin','Display','none','Algorithm','levenberg-marquardt');
		x = lsqnonlin(@f,x,[],[],optim);
		fprintf('final error:	%.02f	(pnp=%.02f,	epi=%.02f)\n', rms(f(x)), rms(f_rpe(x)), rms(f_epi(x)));
	end

	M = vec2mot(x);
	epi = f_epi(x);
	rpe = f_rpe(x);

	if isempty(im), return; end
	
	cmap = uint8(255*jet(64));
	
	if ~isempty(egomo.pnp.uid)
		[rpe,p] = f_rpe(x);
		if ~isfield(egomo.pnp,'val') || isempty(egomo.pnp.val)
			figure, imshow(im); hold on; showMatches(egomo.pnp.uid,egomo.pnp.dst,p,rpe,cmap);
			figure, imshow(im); hold on; showMatches(egomo.pnp.uid,egomo.pnp.dst,p,egomo.pnp.src(:,3),flipdim(cmap,1));
			figure, imshow(im); hold on; showMatches(egomo.pnp.uid,egomo.pnp.dst,p,egomo.pnp.w,flipdim(cmap,1));
		else
			figure, imshow(im); hold on; showMatches(egomo.pnp.uid,egomo.pnp.dst,p,egomo.pnp.val,cmap);
		end
	end

	if ~isfield(egomo.epi,'val') || isempty(egomo.epi.val)
		P0 = egomo.K * [eye(3),zeros(3,1)];
		P1 = egomo.K * M(1:3,:);
		[g,e,~] = triangulatePoints(egomo.epi.src,egomo.epi.dst,P0,P1);
		figure, imshow(im); hold on; showMatches(egomo.epi.uid,egomo.epi.src,egomo.epi.dst,g(:,3),flipdim(cmap,1));
		figure, imshow(im); hold on; showMatches(egomo.epi.uid,egomo.epi.src,egomo.epi.dst,e,cmap);
	else
		figure, imshow(im); hold on; showMatches(egomo.epi.uid,egomo.epi.src,egomo.epi.dst,egomo.epi.val,cmap);
	end
	
	function y = f(x)
		if a_epi > 0, y_epi = a_epi * f_epi(x); else y_epi = []; end
		if a_rpe > 0, y_rpe = a_rpe * f_rpe(x); else y_rpe = []; end
		%y = [y_epi;y_rpe(idx)];
		y = [y_epi;y_rpe];
	end

	function [y,p] = f_rpe(x)
		M = vec2mot(x);
		P = egomo.K*M(1:3,:);
		p = homo2eucl(eucl2homo(egomo.pnp.src) * P');
		y = egomo.pnp.dst - p;
		
		if ~extended, y = egomo.pnp.w .* sqrt(y(:,1).^2 + y(:,2).^2);
		else          y = reshape(bsxfun(@times,y,egomo.pnp.w),[],1);
		end
	end

	function y = f_epi(x)
		M = vec2mot(x);
		F = mot2fmat(egomo.K,M);
		
		if ~extended, y = egomo.epi.w .* sqrt(computeEpipolarError(egomo.epi.src,egomo.epi.dst,F,'sampson'));
		else          y = reshape(bsxfun(@times,computeEpipolarError(egomo.epi.src,egomo.epi.dst,F,'geometric'),egomo.epi.w),[],1);
		end
	end
end

function F = mot2fmat(K,M)
	K_inv = inv(K);
	E = mot2emat(M);
    F = K_inv' * E * K_inv;
end

function showMatches(u,x,y,v,cmap)
	cidx = min(round((v - min(v)) / (prctile(v,99) - min(v)) * (size(cmap,1)-1)) + 1, size(cmap,1));
	for i = 1 : numel(v)
		xi = x(i,:);
		yi = y(i,:);

		plot([xi(1),yi(1)],[xi(2),yi(2)],'-','color',cmap(cidx(i),:));
		plot(xi(1),xi(2),'o','color',cmap(cidx(i),:));
		
		li = [num2str(u(i)) ': ' sprintf('%.02f',v(i))];
		
		if xi(1) > yi(1)
			text(xi(1)+1,xi(2),li,'color',cmap(cidx(i),:));
		else
			text(yi(1)+1,yi(2),li,'color',cmap(cidx(i),:));
		end
	end
end
