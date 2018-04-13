% stx2xml(STX,XML) stores stixels STX to an xml file XML following the format
% specified by the Daimler's 6D-Vision Ground Truth Stixel Dataset[1].
%
% References:
%   [1] http://www.6d-vision.com/ground-truth-stixel-dataset
%
% See also: xml2stx.
%
function stx2xml(stx,xml,stixelWidth)
    if nargin < 3
        if ~isempty(stx), stixelWidth = stx(1).width;
        else              stixelWidth = 0;
        end
    end

    f = fopen(xml,'w');
    fprintf(f,'<?xml version="1.0" encoding="UTF-8" standalone="no" ?>\r\n');
    fprintf(f,'<GtStixel version="1.0" frameNr="1" numStixels="%d" stixelWidth="%d">\r\n',numel(stx),stixelWidth);

    for i = 1 : numel(stx)
        u = stx(i).x+stx(i).width/2-1;
        b = stx(i).y+stx(i).height-1;
        fprintf(f,'  <Stixel id="%d" u="%.06f" vB="%.06f" vT="%.06f" disp="%.06f" />\r\n',i-1,u-1,b-1,stx(i).y-1,stx(i).d);
    end
    
    fprintf(f,'</GtStixel>\r\n');
    fclose(f);
end