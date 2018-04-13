% CAM2RECT derives rectified patameters from two cameras in canonical geometry
function rect = cam2rect(pri,sec)
    rect.fu = pri.K(1,1);
    rect.fv = pri.K(2,2);
    rect.uc = pri.K(1,3);
    rect.vc = pri.K(2,3);

    % verify that both cameras are rectified together
    assert(sec.K(1,1) == rect.fu && sec.K(2,2) == rect.fv);
    assert(sec.K(1,3) == rect.uc && sec.K(2,3) == rect.vc);
    
    % find the extrinsics
    E = sec.E * invmot(pri.E);
    R = E(1:3,1:3);
    t = E(1:3,4);
    
    % verify both cameras are looking toward the same direction
    assert(all(all(abs(R - eye(3)) < 1e-5)));

    % make sure both cameras are collinear
    % assert(t(2) == 0 && t(3) == 0) % THIS DOESN'T WORK IN TRINOCULAR/MULTIOCULAR CASE

    % get the baseline
    rect.b = t(1);
end