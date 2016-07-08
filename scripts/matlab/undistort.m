% undistort transforms points from an distorted image plane to an ideal one
function x = undistort(x,cam)
    if isa(cam.ParamsObject,'cameraParameters')
        x = undistortPoints(x,cam.ParamsObject); % use the built-in function
    else
        % TODO: finish this for undistortion without the help of
        %  Computer Vision toolbox
        error 'not implemented yet :('
    end
end
