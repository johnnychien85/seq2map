function demo_TwoViews
    K = [720,0,320;0,720,240;0,0,1];
    imageSize = [480,640];

    frm = makeFrame(imageSize(1),imageSize(2));
    frm = frm * inv(K)';

    M0 = eye(4);
    M0(1:3,1:3) = angle2dcm(0,deg2rad(-45),0)';
    M0(1:3,4)   = M0(1:3,1:3) * [ 2,0,0]';

	M1 = eye(4);
	M1(1:3,1:3) = angle2dcm(0,deg2rad(45),0)';
	M1(1:3,4)   = M1(1:3,1:3) * [-2,0,0]';
    
    figure; hold on; axis equal; grid on;
    drawFrame(frm,M0);
    drawFrame(frm,M1);
    
    x = [0,0,2];
    drawProj(M0,x);
    drawProj(M1,x);

    plot3(x(1),x(2),x(3),'ro','markersize',8,'MarkerFaceColor',[1,0,0]);
end

function frm = makeFrame(h,w)
    frm = [0,0,1;w,0,1;w,h,1;0,h,1];
end

function drawFrame(frm,pose)
    pose = inv(pose);
    frm(end+1,:) = [0,0,0];
    frm(:,end+1) = 1;
    frm = frm * pose(1:3,:)';
    fill3(frm(1:4,1)',frm(1:4,2)',frm(1:4,3)',[1,1,1]);
    alpha(0.2);
    for i = 1 : 4
        plot3([frm(end,1),frm(i,1)],[frm(end,2),frm(i,2)],[frm(end,3),frm(i,3)],'k-');
    end
end

function drawProj(pose,x)
    y = [x,1] * pose(1:3,:)';
    y = y / y(3);
    
    pose = inv(pose);
    y = [y,1] * pose(1:3,:)';
    c = pose(1:3,4);
    
    plot3([x(1),y(1),c(1)],[x(2),y(2),c(2)],[x(3),y(3),c(3)],'k-','LineWidth',2);
    plot3(y(1),y(2),y(3),'ko','markersize',8,'MarkerFaceColor',[0,0,0]);
end