function seq = loadSequence(seqPath)
    if nargin < 1, seqPath = uigetdir(); end

    fprintf('Loading VO + LiDAR sequence [%s]..\n', seqPath);

    kittiIMUPath = fullfile(seqPath, 'oxts');

    if exist(kittiIMUPath, 'dir' ) > 0, seq = loadSequenceKITTI(seqPath);
    else                                seq = loadSequenceVG   (seqPath); end
    
    %playSequence(seq);
end

%
% Load a sequence downloaded from http://www.cvlibs.net/datasets/kitti/raw_data.php
% both [unsynced+unrectified data] and [synced+rectified data] sequences are supported.
%
function seq = loadSequenceKITTI(seqPath)
    fprintf('..this sequence was grabbed by the KITTI car!!\n');

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
        cam(i).E(1:3,4) = cam(i).E(1:3,4) * 100;
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
    lid(1).E(1:3,4) = lid(1).E(1:3,4) * 100;

    % check the number of LiDAR frames
    if ~isempty(lid(1).DataFiles), assert(numel(lid(1).DataFiles) == frames); end

    % load IMU motion and convert it to the left camera's coordinate system
   	if exist(paths.imuPath, 'dir') > 0
		cam(1).M = zeros(4,4,frames);
        oxts = loadOxtsliteData(seqPath);
        pose = convertOxtsToPose(oxts);

        assert(numel(pose) == frames);

        for t = 1 : frames, cam(1).M(:,:,t) = imu2cam * inv(pose{t}) * cam2imu; end
        cam(1).M(1:3,4,:) = cam(1).M(1:3,4,:) * 100;
    end
    
    seq = struct(...
        'Grabber',  'KITTI',...
        'Paths',    paths,  ...
        'cam',      cam,    ...
        'lid',      lid     ...
    );
end

%
% Sequence loader exclusive for VGLAB
%
function seq = loadSequenceVG(seqPath)
    fprintf('..perhaps this sequence was grabbed by vggrab (fingers crossed..)\n');

    paths = struct(                                          ...
        'seqPath', seqPath,                                  ...
        'calPath', fullfile(seqPath, 'cal', 'cam.m'),        ...
        'ldbPath', fullfile(seqPath, 'cal', 'lidar-db.xml'), ...
        'extPath', fullfile(seqPath, 'cal', 'extrinsics.m'), ...
        'motPath', fullfile(seqPath, 'mot.mat'),             ...
        'imgRoot', fullfile(seqPath, 'rect'),                ...
        'lidRoot', fullfile(seqPath, 'lidar')                ...
    );

    % Load camera parameters
    run(paths.calPath);

    % Do some sanity checks..
    assert(exist('P','var') == 1 && exist('R','var') == 1 && exist('imageSize','var'));
    assert(size(P,1) == 3 && size(P,2) == 4 && size(R,1) == 3 && size(R,2) == 3);
    assert(size(P,3) == size(R,3));
    cams = size(P,3);

    % Fill in the parameters and locate image files for each camera
    for i = 1 : cams
        cam(i) = makeCameraStruct();
        cam(i).PixelClass = 'uint8';
        cam(i).K = P(1:3,1:3,i);
        cam(i).E = rect2extrinsics(P(:,:,i),R(:,:,i));
        cam(i).P = cam(i).K * cam(i).E(1:3,:);
        cam(i).ParamsObject = cam2obj(cam(i));
		cam(i).ImageSize = imageSize(i,:);
    end
    fprintf('..camera intrinsics loaded, %d camera(s) found\n', cams);

    [imgs,found] = findImages(paths.imgRoot,cams,'*.png');
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
        'Paths',    paths,      ...
        'Grabber',  'vggrab',   ...
        'cam',      cam,        ...
        'lid',      lid         ...
    );
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
        'M', zeros(4,4,0),   ... % 4-by-4-by-k motion matrix
        'ImageSize',  [0,0], ... % rows and cols of image
        'ImageFiles', [],    ... % paths to the image files
        'Demosaic',   [],    ... % demosaic spec if required
        'PixelClass', '',    ... % MATLAB class name of image pixel
        'ParamsObject', []   ... % MATLAB camera parameters object if supported
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
    assert(numel(subs) <= cams, 'too many image sub folders!!');

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
        dirPath = fileparts(pattern); % take into account the path contained in the pattern if there exists one
        if strcmp(dirPath, '..'), dirPath = ''; end
        f = [f; cellfun(@(x)fullfile(root,dirPath,x),{files(:).name}','UniformOutput',false)];
    end
end

% References
% [1] Geiger's IJPR2013 paper: http://www.cvlibs.net/publications/Geiger2013IJRR.pdf