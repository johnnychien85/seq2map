function [ER,Et,Tr,Ta,milestones,fig] = compareMotion(m0,m1,lbl,mks,tn,unit)
	if ~iscell(m1), m1 = {m1}; end;
	if nargin < 3 || isempty(lbl), for k = 0 : numel(m1), lbl{k+1} = sprintf('M%d', k); end; end
	if nargin < 4 || isempty(mks), for k = 0 : numel(m1), mks{k+1} = '-';               end; end
    if nargin < 5 || isinf(tn),    tn = min(cellfun(@(x)size(x,3),m1));   end
    if nargin < 6, unit = 1;        end

	% divie into 10 equally spaced intervals
	n = 10;
	m = numel(m1);

    m0 = m0(:,:,1:tn);
    m0(1:3,4,:) = m0(1:3,4,:) * unit;
    for k = 1 : m
        m1{k}(1:3,4,:) = m1{k}(1:3,4,:) * unit;
        m1{k} = m1{k}(:,:,1:tn);
    end
    
	segs = m2segs(m0);

	milestones = linspace(0,sum(segs),n+1);
	milestones = milestones(2:end);

	ER = cell(m,n);
	Et = cell(m,n);
	Tr = zeros(m,tn-1);
	Ta = zeros(m,tn-1);
	C  = lines(m);
	la = 0;

	for t0 = 1 : tn - 1
		for s = 1 : n
			[t1,l] = findTestEnd(t0,segs,milestones(s));
			if isinf(t1), continue; end;

			m0j = m0(:,:,t0) * inv(m0(:,:,t1));
			
			for k = 1 : m
				m1j = m1{k}(:,:,t1) * inv(m1{k}(:,:,t0));
				[er,et] = m2e(m1j*m0j);
				lk = norm(m0j(1:3,4));
				
				ER{k,s}(end+1) = er / lk;
				Et{k,s}(end+1) = et / lk * 100;
			end
		end

		t1 = t0 + 1;
		for k = 1 : m
			m0j = m0(:,:,t0) * inv(m0(:,:,t1));
			m1j = m1{k}(:,:,t1) * inv(m1{k}(:,:,t0));
			[~,et] = m2e(m1j*m0j);
			Tr(k,t0) = norm(et);% / norm(m0j(1:3,4)) * 100;
			
			m0j = m0(:,:,1) * inv(m0(:,:,t1));
			m1j = m1{k}(:,:,t1) * inv(m1{k}(:,:,1));
			[~,et] = m2e(m1j*m0j);
			Ta(k,t0) = norm(et);
		end

		m0j = m0(:,:,t0) * inv(m0(:,:,t1));
		la = la + norm(m0j(1:3,4));
	end

	Ta = Ta / la * 100;

	ER = cellfun(@mean,ER);
	Et = cellfun(@mean,Et);

	%mean(Et,2);
	
	figure; subplot(2,1,1); grid on; hold on; xlabel 'Frame'; ylabel 'Inter-frame Drift (%)';
	for k = 1 : m, plot(1:tn-1,Tr(k,:),mks{k},'color',C(k,:),'linewidth',1); end
	legend(lbl{2:end},'Location','EastOutside');
	xlim([1 tn-1]);

	subplot(2,1,2); grid on; hold on; xlabel 'Frame'; ylabel 'Accumulated Drift (%)';
	for k = 1 : m, plot(1:tn-1,Ta(k,:),mks{k},'color',C(k,:),'linewidth',2); end
	legend(lbl{2:end},'Location','EastOutside');
	xlim([1 tn-1]);
  
	fig = figure;
	subplot(1,2,1); hold on; grid on;
	% title(sprintf('m0=%.02f%%', mean(Et)));
	xlabel 'Path Length (m)';
	ylabel 'Translation Error (%)';
	xlim([milestones(1),milestones(end)]);
	for k = 1 : m, 	plot(milestones,Et(k,:),'-','color',C(k,:),'linewidth',2); end
	legend(lbl{2:end});

	subplot(1,2,2); hold on; grid on;
	% title(sprintf('m0=%.02f deg/m', mean(ER)));
	xlabel 'Path Length (m)';
	ylabel 'Rotation Error (deg/m)';
	xlim([milestones(1),milestones(end)]);
	for k = 1 : m, 	plot(milestones,ER(k,:),'-','color',C(k,:),'linewidth',2); end
	legend(lbl{2:end});
	
	% subplot(1,3,3); hold on; grid on; axis equal;
	figure, hold on; grid on; axis equal;
	xlabel 'x (m)', ylabel 'y (m)', zlabel 'z (m)';
	view([0,1,0]);
	drawMotion(m0(:,:,1:tn), [0,0,0], mks{1});
	for k = 1 : m
		drawMotion(m1{k}(:,:,1:tn), C(k,:), mks{k+1});
	end
	legend(lbl, 'Location','NorthEast');
end

function segs = m2segs(m)
	tn = size(m,3);
	segs = zeros(tn-1,1);
	for t = 1 : tn - 1
		mt = m(:,:,t+1) * inv(m(:,:,t));
		segs(t) = norm(mt(1:3,4));
	end;
end

function [t,l] = findTestEnd(t0,segs,milestone)
	asum = cumsum(segs(t0:end));
	idx = find(asum >= milestone);
	if isempty(idx)
		t = inf;
		l = inf;
	else
		t = t0 + idx(1);
		l = asum(idx(1));
	end
end

function [er,et] = m2e(dm)
	qr = dcm2quat(dm(1:3,1:3));
	er = abs(acosd(qr(1)) * 2);
	et = norm(dm(1:3,4));
end
