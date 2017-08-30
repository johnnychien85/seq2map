function seq = raw2seq(seqPath)
    if nargin < 1, seqPath = uigetdir(); end

    fprintf('Loading VO + LiDAR sequence [%s]..\n', seqPath);

	kittiCalibPath = fullfile(seqPath, 'calib.txt');
    kittiIMUPath   = fullfile(seqPath, 'oxts');

	if     exist(kittiCalibPath, 'file') > 0, seq = loadSequenceKITTI   (seqPath);
    elseif exist(kittiIMUPath,   'dir' ) > 0, seq = loadSequenceKITTIRaw(seqPath);
    else                                      seq = loadSequenceVG      (seqPath); end

	featuresPath = fullfile(seqPath,'kpt');
	if exist(featuresPath, 'dir') > 0
		fprintf('Possible existence of feature files detected!\n');
		seq = locateFeatures(seq,featuresPath);
	end
    
    dpmapsPath = fullfile(seqPath,'dpm');
    if exist(dpmapsPath,'dir') > 0
        fprintf('Possible existence of disparity maps detected!\n');
        dmax = input('Please specify dmax: ');
        seq = locateDpmaps(seq,dpmapsPath,dmax);
    end
end

%
% Load a sequence downloaded from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
% An odometry sequence comes with stereo images and optionally GPS/IMU readings
% and LiDAR point clouds. The sequences are supposed to be structured in this way:
%  <DATASET_ROOT>
%   + 00 <----------- seqPath
%   |  + image_0
%   |  | + 000000.png
%   |  | + 000001.png
%   |  | .
%   |  | .
%   |  + image_1
%   |  + image_2
%   |  + image_3
%   |  + calib.txt
%   |  \ times.txt
%   + 01
%   + 02
%   .
%   .
%   \ 21
%
% and the ground truth poses are supposed to be stored at <DATASET_ROOT>/../poses/
% where contains the downloaded ground truth poses 00.txt, 01.txt, ..., 10.txt.
%
function seq = loadSequenceKITTI(seqPath)
    fprintf('..this sequence is from KITTI Odometry dataset!!\n');

	[~,seqNum] = fileparts(seqPath);
	
	poseFile = [seqNum '.txt'];
	
    paths = struct(                                               ...
        'seqPath', seqPath,                                       ...
		'calPath', fullfile(seqPath,'calib.txt'),                 ...
        'imuPath', fullfile(seqPath,'../','../','poses',poseFile),...
        'imgPath', fullfile(seqPath,'image_*'),                   ...
        'lidPath', fullfile(seqPath,'velodyne')                   ...
    );
	
	%
	% 1. Load camera calibration
	%
	if exist(paths.calPath,'file') == 00
		error 'missing calibration!!';
	end

	f = fopen(paths.calPath);
	[cal,toks] = fscanf(f,'%s %f %f %f %f %f %f %f %f %f %f %f %f');
	fclose(f);
	
	assert(toks - 1 == 64);

	cal = reshape(cal,15,[]);
	cal = reshape(cal(4:end,:),4,3,[]);

	% check if the 3rd and the 4th colour cameras are used
	d = dir(paths.imgPath);
	cams = numel(d);

	if     cams == 2, fprintf('..2 greyscale cameras found!\n');
	elseif cams == 4, fprintf('..2 greyscale + 2 colour cameras found!!\n');
	else,             error 'something wrong with the number of image folders.';
	end

	for k = 1 : cams
		cam(k) = makeCameraStruct();
		cam(k).PixelClass = 'uint8';
		cam(k).P = cal(:,:,k)';
		
		if k == 1 % KITTI uses the first camera as the referenced frame
			cam(k).K = cam(k).P(1:3,1:3);
			K_inv = inv(cam(1).K);
		else
			cam(k).K = cam(1).K;
			cam(k).E(1:3,:) = K_inv * cam(k).P(1:3,:);
		end
	end

	%
	% 2. Locate image files
	%
    [imgs,found] = findImages(paths.imgPath, cams, '*.png');
    frames = size(imgs, 2);
    for k = 1 : cams, cam(k).ImageFiles = imgs(k,:); end
    fprintf('..image data of %d camera(s) and %d frames located\n', found, frames);
	
	%
	% 3. LiDAR..
	%
	lid2cam = cal(:,:,5)';
    lid(1) = makeLiDARStruct();
    lid(1).Format = 'XYZI'; % KITTI stores LiDAR readings as (x,y,z,intensity) tuples
    lid(1).DataFiles = fdir(paths.lidPath, '*.bin');
    lid(1).E = [lid2cam;0,0,0,1];
	
    if ~isempty(lid(1).DataFiles)
        fprintf('..found %d Velodyne LiDAR data\n',numel(lid(1).DataFiles));
    end
    
    %
    % 4. Motion
    %
    if exist(paths.imuPath,'file') > 0
        cam(1).M = txt2mot(paths.imuPath);
        for k = 1 : size(cam(1).M,3)
            cam(1).M(:,:,k) = invmot(cam(1).M(:,:,k));
        end
        
        fprintf('..pose loaded from %s\n',paths.imuPath);
    end
    
	seq = struct(...
        'Grabber',  'KITTI',...
        'Paths',    paths,  ...
        'cam',      cam,    ...
        'lid',      lid     ...
    );
end

%
% Load a sequence downloaded from http://www.cvlibs.net/datasets/kitti/raw_data.php
% both "unsynced+unrectified data" and "synced+rectified data" sequences are supported.
%  <DATASET_ROOT>
%   + <DATE_LABEL>
%   |  + <SEQUENCE_LABEL>[_sync] <--------- seqPath
%   |  |  + image_00
%   |  |  |  + data
%   |  |  |  |  + 0000000000.png
%   |  |  |  |  + 0000000001.png
%   |  |  |  |  .
%   |  |  |  |  .
%   |  |  |  \ timestamp.txt  
%   |  |  + image_01
%   |  |  + image_02
%   |  |  + image_03
%   |  |  + oxts
%   |  |  \ velodyne_points
%
function seq = loadSequenceKITTIRaw(seqPath)
    fprintf('..this RAW sequence was grabbed by the KITTI car!!\n');

    paths = struct( ...
        'seqPath', seqPath,                                   ...
        'sdkPath', fullfile('..','..','3rdparties','kitti','devkit_raw_data','devkit'),...
        'calRoot', fullfile(seqPath,'../'),                   ...
        'cam2cam', 'calib_cam_to_cam.txt',                    ...
        'imu2lid', 'calib_imu_to_velo.txt',                   ...
        'lid2cam', 'calib_velo_to_cam.txt',                   ...
        'imgPath', fullfile(seqPath,'image_*','data'),        ...
        'lidPath', fullfile(seqPath,'velodyne_points','data'),...
        'imuPath', fullfile(seqPath,'oxts','data')            ...
    );

    % Locate the required SDK files if not exist
    if exist('loadOxtsliteData.m','file') == 0
        if exist(paths.sdkPath,'dir') == 0
            prompt = 'Please locate the devkit folder of the KITTI''s Raw Data SDK';
            paths.sdkPath = uigetdir([],prompt);
        end
        paths.sdkPath = fullfile(paths.sdkPath,'matlab');
        addpath(paths.sdkPath);
    end

    % Locate the calibration files if not exist
    if exist(fullfile(paths.calRoot,paths.cam2cam),'file') == 0
        prompt = 'Please locate the folder containing calibration files (e.g. calib_cam_to_cam.txt)';
        paths.calRoot = uigetdir([],prompt);
    end

    % Determine if the sequence has been rectified or not.
    %  Note KITTI provides both raw and rectified sequences. Hints are needed
    %  to tell if the sequence being loaded is raw or not.
    [~,seqName,~] = fileparts(seqPath);
    rectified = strfind(seqName, '_sync'); % the "hint" we use here

    paths.cam2cam = fullfile(paths.calRoot, paths.cam2cam);
    paths.imu2lid = fullfile(paths.calRoot, paths.imu2lid);
    paths.lid2cam = fullfile(paths.calRoot, paths.lid2cam);

    % load the calibrated intrinsics & extrinsics first
    calib   = loadCalibrationCamToCam(paths.cam2cam);
    imu2lid = loadCalibrationRigid(paths.imu2lid);
    lid2cam = loadCalibrationRigid(paths.lid2cam);
    imu2cam = lid2cam * imu2lid; % note the order multiplication
	cam2imu = inv(imu2cam);

    cams = numel(calib.K);

    % fill in the parameters and locate image files for each camera
    for i = 1 : cams
        cam(i) = makeCameraStruct();
        cam(i).PixelClass = 'uint8';

        if ~rectified % unrectified case
            cam(i).ImageSize = flipdim(calib.S{i},2); % do flipping so we have [rows cols]
            cam(i).K = calib.K{i};
            cam(i).D = calib.D{i}';
            cam(i).E = [calib.R{i}, calib.T{i}];
        else % rectified case requires further computation
            cam(i).ImageSize = flipdim(calib.S_rect{i},2);
            cam(i).K = calib.P_rect{i}(1:3,1:3);
            cam(i).E = rect2extrinsics(calib.P_rect{i}, calib.R_rect{1}); % see Eq (5) in Geiger's paper[1]
            cam(i).P = cam(i).K * cam(i).E(1:3,:);
        end
        cam(i).ParamsObject = cam2obj(cam(i));
        % cam(i).E(1:3,4) = cam(i).E(1:3,4) * 100;
    end
    if rectified, fprintf('..camera intrinsics loaded, %d rectified camera(s) found\n',   cams);
    else          fprintf('..camera intrinsics loaded, %d UNRECTIFIED camera(s) found\n', cams); end

    % locate image files
    [imgRoot,imgFile,~] = fileparts(paths.imgPath);
    [imgs,found] = findImages(imgRoot, cams, fullfile(imgFile, '*.png'));
    frames = size(imgs, 2);
    for i = 1 : cams, cam(i).ImageFiles = imgs(i,:); end
    fprintf('..image data of %d camera(s) and %d frames located\n', found, frames);

    % locate LiDAR data
    lid(1) = makeLiDARStruct();
    lid(1).Format = 'XYZI'; % KITTI stores LiDAR readings as (x,y,z,intensity) tuples
    lid(1).DataFiles = fdir(paths.lidPath, '*.bin');
    lid(1).E = lid2cam;

    % check the number of LiDAR frames
    if ~isempty(lid(1).DataFiles), assert(numel(lid(1).DataFiles) == frames); end

    % load IMU motion and convert it to the left camera's coordinate system
   	if exist(paths.imuPath,'dir') > 0
		cam(1).M = zeros(4,4,frames);
        oxts = loadOxtsliteData(seqPath);
        pose = convertOxtsToPose(oxts);
        assert(numel(pose) == frames);
        for t = 1 : frames
            cam(1).M(:,:,t) = imu2cam * inv(pose{t}) * cam2imu;
        end
    end
    
    seq = struct(...
        'Grabber','KITTI-RAW',...
        'Paths',  paths,      ...
        'Frames', frames,     ...
        'cam',    cam,        ...
        'lid',    lid         ...
    );
end

%
% Sequence loader exclusive for VGLAB
%
function seq = loadSequenceVG(seqPath)
    fprintf('..perhaps this sequence was grabbed by vggrab (fingers crossed..)\n');

    imgRoot = fullfile(seqPath,'rect');
    rectified = exist(imgRoot,'dir') ~= 0;

    if ~rectified, imgRoot = fullfile(seqPath,'raw'); end

    paths = struct(                                       ...
        'seqPath', seqPath,                               ...
        'calPath', fullfile(seqPath,'cal','camvtx.m'),    ...
        'ldbPath', fullfile(seqPath,'cal','lidar-db.xml'),...
        'extPath', fullfile(seqPath,'cal','extrinsics.m'),...
        'motPath', fullfile(seqPath,'mot.mat'),           ...
        'imgRoot', imgRoot,                               ...
        'lidRoot', fullfile(seqPath,'lidar')              ...
    );

    % Load camera parameters
    run(paths.calPath);

    % Do some sanity checks..
    cams = numel(vtx);

    % Fill in the parameters and locate image files for each camera
    for i = 1 : cams
        cam(i) = makeCameraStruct();
        cam(i).PixelClass = 'uint8';
        
        if ~rectified            
            cam(i).K = vtx(i).K;
            cam(i).P = vtx(i).P;
            cam(i).E = [vtx(i).R, vtx(i).t; 0, 0, 0, 1];
            cam(i).ImageSize = vtx(i).S;
        else
            cam(i).K = vtx(1).P_rect(1:3,1:3);
            cam(i).P = vtx(i).P_rect;
            cam(i).E = inv(cam(i).K) * cam(i).P;
            cam(i).ImageSize = vtx(i).S_rect;
        end
        cam(i).ParamsObject = cam2obj(cam(i));  
    end
    fprintf('..camera intrinsics loaded, %d camera(s) found\n', cams);

    [imgs,found] = findImages(paths.imgRoot,cams,{'*.pgm','*.png'});
    frames = size(imgs,2);
    for i = 1 : cams, cam(i).ImageFiles = imgs(i,:); end
    fprintf('..image data of %d camera(s) and %d frames located\n', found, frames);

    % Load camera motion if available
    if exist(paths.motPath) == 2
        load   (paths.motPath);
        assert (exist('M','var') == 1);
        assert (size(M,1) == 4 && size(M,2) == 4 && size(M,3) >= frames);
        cam(1).M = M;
        fprintf('..camera motion loaded, %d frames found\n', size(M,3));
    end

    % Load LiDAR intrinsics
    if exist(paths.ldbPath) == 2
        % Load extrinsics
        if exist(paths.extPath) == 2
            run (paths.extPath);
            assert(exist('E','var') == 1 && all(size(E) == [4,4]));
            fprintf('..LiDAR-camera extrinsics loaded\n');
        else
            E = [1,0,0,0;0,0,-1,0;0,1,0,0;0,0,0,1];
        end

        lid(1) = makeLiDARStruct();
        lid(1).Format = 'VLD64';
        lid(1).AT = VLDParams2AT(VLDLoadParams(paths.ldbPath));
        lid(1).DataFiles = fdir(paths.lidRoot,'*.dat');
        lid(1).E = E;

        fprintf('..LiDAR intrinsics parsed, %d frames found\n', numel(lid(1).DataFiles));
    else
        fprintf('..LiDAR data skipped due to missing intrinsics %s\n', paths.ldbPath);
        lid = [];
    end

    seq = struct(...
        'Paths',   paths,    ...
        'Grabber', 'vggrab', ...
        'Frames',  frames,   ...
        'cam',     cam,      ...
        'lid',     lid       ...
    );
end

function seq = locateFeatures(seq,featuresPath)
	subs = subdirs(featuresPath);
	
	if numel(subs) == 0
		fprintf('..nothing found, give up\n');
		return;
	elseif numel(subs) > 1
		fprintf('..more than 1 folders found in %s, manual selection required\n',featuresPath);
		featuresPath = uigetdir(featuresPath);
	else
		featuresPath = fullfile(featuresPath,subs(1).name);
	end

	subs = subdirs(featuresPath);
	cams = numel(seq.cam);
	
	if numel(subs) > cams
		fprintf('..%d folder(s) found while we have only %d camera(s), gonna give up!\n',numel(subs),cams);
		return;
	end

	for k = 1 : numel(subs)
		seq.cam(k).FeatureFiles = fdir(fullfile(featuresPath,subs(k).name),'*.*');
		features = numel(seq.cam(k).FeatureFiles);
		images   = numel(seq.cam(k).ImageFiles);

		assert(images == features);
		
		fprintf('..%d feature file(s) found for camera %d\n', numel(seq.cam(k).FeatureFiles),k);
	end
end

function seq = locateDpmaps(seq,dpmapsPath,dmax)
    cams = numel(seq.cam);
    i = 0;

    for c0 = 1 : cams
        for c1 = 1 : cams
            if c0 == c1, continue; end
            
            dpmapName = sprintf('d%d%d',c0-1,c1-1);
            dpmapPath = fullfile(dpmapsPath,dpmapName);
                       
            if exist(dpmapPath,'dir') == 0, continue; end
            
            fprintf('..disparity maps (%d,%d) found!!\n',c0,c1);
            i = i + 1;
            
            seq.dpm(i).Name = dpmapName;
            seq.dpm(i).DisparityFiles = fdir(dpmapPath,'*.*');
            seq.dpm(i).c0 = c0;
            seq.dpm(i).c1 = c1;
            seq.dpm(i).dmax = dmax;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Create an empty structure containing imaging parameters
%  essential for I/O and geometric operations
%
function cam = makeCameraStruct()
    cam = struct(        ...
        'K', eye(3),     ... % 3-by-3 camera matrix
        'D', zeros(5,1), ... % 5-by-1 distortion coeffs.: kappa1 kappa2 p1 p2 and kappa3
        'E', eye(4),     ... % 4-by-4 pose matrix
        'P', [eye(3),zeros(3,1)], ... % 3-by-4 projection matrix
        'M', zeros(4,4,0),     ... % 4-by-4-by-k motion matrix
        'ImageSize',    [0,0], ... % rows and cols of image
        'ImageFiles',   [],    ... % paths to the image files
		'FeatureFiles', [],    ... % paths to the stored feature files
        'Demosaic',     [],    ... % demosaic spec if required
        'PixelClass',   '',    ... % MATLAB class name of image pixel
        'ParamsObject', []     ... % MATLAB camera parameters object if supported
    );
end

%
% Create a MATLAB camera parameters object from a given cam structure
%
function obj = cam2obj(cam)
    if exist('cameraParameters', 'class') == 0 % we need Computer Vision toolbox!!
        obj = [];
        return;
    end

    % convert rotation matrix to vector representation
    a = vrrotmat2vec(cam.E(1:3,1:3));
    a = a(1:3) * a(4);

    obj = cameraParameters(...
        'IntrinsicMatrix',  cam.K',           ... % intrinsics (fu, fv, uc, vc)
        'RadialDistortion', cam.D([1,2,5]),   ... % kappa1, kappa2, and kappa3
        'TangentialDistortion', cam.D(3:4),   ... % p1 and p2
        'NumRadialDistortionCoefficients', 3, ... % kappa3 included
        'RotationVectors', a,                 ... % Rvec
        'TranslationVectors', cam.E(1:3,4)',  ... % tvec
        'WorldUnit', 'cm'                     ... % we always use 'cm'
    );
end

%
% Create an empty structure containing LiDAR parameters
%
function lid = makeLiDARStruct()
    lid = struct( ...
        'Format', '',    ... % format of the LiDAR data
        'DataFiles', [], ... % paths to the LiDAR data
        'E', eye(4),     ... % 4-by-4 pose matrix
        'AT', zeros(0,6) ... % n-by-6 intrinsics used by Velodyne LiDARs
    );
end

%
% Find extrinsics of a recitified camera from it's rectifying projection and rotation matrices
%
function E = rect2extrinsics(P,R)
    K = P(1:3,1:3);
    t = inv(K) * P(1:3,4);
    E = [R,t;0,0,0,1];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Locate image files
%
function [imgs,found] = findImages(imgRoot,cams,patterns)
    subs = subdirs(imgRoot);
    assert(numel(subs) == cams, 'the number of image folders and the number of cameras mismatch!!');

    % if imgRoot specifies a pattern of folder name instead of the path to the
    %  root image directory, then we go up to the parent folder
    if strfind(imgRoot, '*'), imgRoot = fileparts(imgRoot); end

    for i = 1 : numel(subs)
        imgs_i = fdir(fullfile(imgRoot,subs(i).name),patterns);

        if i == 1, imgs = cell(cams,numel(imgs_i));
        else       assert(numel(imgs_i) == size(imgs,2)); end

        imgs(i,:) = imgs_i';
    end
    
    found = min(numel(subs), cams);
end

%
% List sub-directories
%
function d = subdirs(folder)
    d = dir(folder);
    d = d(arrayfun(@(x) x.isdir && ~ismember(x.name, {'.','..'}), d));
end

%
% Search for files in a directory that match one of the given name patterns
%
function f = fdir(root, patterns)
    if ~iscell(patterns), patterns = {patterns}; end
    f = cell(0,1);
    for i = 1 : numel(patterns)
        pattern = patterns{i};
        files   = dir(fullfile(root,pattern));
		files   = files(arrayfun(@(x) ~ismember(x.name,{'.','..'}),files)); % remove ".." and "." entries
        dirPath = fileparts(pattern); % take into account the path contained in the pattern if there exists one
        if strcmp(dirPath, '..'), dirPath = ''; end
        f = [f; cellfun(@(x)fullfile(root,dirPath,x),{files(:).name}','UniformOutput',false)];
    end
end

% References
% [1] Geiger's IJPR2013 paper: http://www.cvlibs.net/publications/Geiger2013IJRR.pdf