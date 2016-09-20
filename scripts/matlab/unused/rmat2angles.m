% RMAT2ANGLES converts a rotation matrix to euler angles which define three
% consequent rotations respectively about x-, y-, and z-axis.
% 
% Source of formula: http://planning.cs.uiuc.edu/node103.html
%
function [x,y,z] = rmat2angles(R)
    z = atan( R(2,1) / R(1,1));
    y = atan(-R(3,1) / sqrt(R(3,2)^2+R(3,3)^2));
    x = atan( R(3,2) / R(3,3));
end
