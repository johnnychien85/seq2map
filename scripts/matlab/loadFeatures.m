function F = loadFeatures(cam,frame)
	
	filename = cam.FeatureFiles{frame};

	f = fopen(filename,'r');
	magic = char(fread(f,[1,8],'char'));

	if ~strcmp(magic,'IMKPTDSC')
		fclose(f);
		error 'magic number check failed';
	end

	keypointFormat = fscanf(f,'%s',1);
	descriptorNorm = fscanf(f,'%s',1);
	descriptorType = fscanf(f,'%s',1);

	if isempty(keypointFormat) || isempty(descriptorType) || isempty(descriptorNorm)
		fclose(f);
		error 'sanity check failed';
	end

	fseek(f,1,0);

	numFeatures    = fread(f,1,'int');
	descriptorSize = fread(f,1,'int');
	descriptorType = cv2matlabType(descriptorType);

	F = struct(...
		'NumOfFeatures', numFeatures,                     ...
		'Metric',        normType2metric(descriptorNorm), ...
		'KP',            zeros(numFeatures,2,'single'),   ...
		'V',             zeros(numFeatures,descriptorSize)...
	);

	for i = 1 : numFeatures
		F.KP(i,:) = fread(f,2,'single'); % key point location
		score = fread(f,1,'single'); % detector response, unused
		scale = fread(f,1,'int');    % feature's octave, unused
		angle = fread(f,1,'single'); % feature's orientation, unused
		radii = fread(f,1,'single'); % feature's neighbourhood radius, unused
	end

	% load feature descriptors
	precision = [descriptorType '=>' descriptorType];
	F.V = fread(f,[descriptorSize,numFeatures],precision)';
	
	fclose(f);
end

function type = cv2matlabType(type)
	switch type
		case '8U',  type = 'uchar';
		case '8S',  type = 'char';
		case '16U', type = 'ushort';
		case '16S', type = 'short';
		case '32S', type = 'int';
		case '32F', type = 'single';
		case '64F', type = 'double';
		otherwise,  error 'unknmown OpenCV type!';
	end
end

function metric = normType2metric(normType)
	switch normType
		case {'INF','L1'},   		 metric = 'SAD';
		case {'L2','L2SQR'}, 		 metric = 'SSD';
		case {'HAMMING','HAMMING2'}, metric = 'HAMMING';
		otherwise 					 error 'unknown norm type!';
	end
end