function x = raw2mat(from,m,n,s)
    f = fopen(from,'r');
    if nargin < 2 || isstr(m)
      if nargin < 2, x = loadCvMat(f);
      else           x = loadCvMat(f,m);
      end
    else
        if nargin < 3, s = 'int16'; end;
        x = fread(f,[n m],s)';
    end
    fclose(f);
end

function x = loadCvMat(f,s)
    magic = char(fread(f,[1,5],'char'));
    if ~strcmp(magic,'CVMAT')
		error 'magic number check failed';
	end

    type = cv2matlabType(fscanf(f,'%s',1));
    
    if isempty(type) && nargin < 2
      error 'USR matrix requires explicit MATLAB type spec';
    end

    if nargin >= 2
      if ~isempty(type)
        warning('self-specified type %s is  overriden by %s',type,s);
      end
      type = s;
    end
    
    fseek(f,1,0);    

    m = fread(f,1,'int');
    n = fread(f,1,'int');
    k = fread(f,1,'int');
    
    x = permute(reshape(fread(f,m*n*k,['*' type]),[k,n,m]),[3,2,1]);
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
        case 'USR', type = '';
		otherwise,  error 'unknmown OpenCV type!';
        %otherwise, type = 'uint64';
	end
end