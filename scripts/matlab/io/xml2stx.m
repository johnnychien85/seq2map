% STX = xml2stx(XML) reads stixels STX from file XML following the format
% specified by the Daimler's 6D-Vision Ground Truth Stixel Dataset[1].
%
% References:
%   [1] http://www.6d-vision.com/ground-truth-stixel-dataset
%
% See also: stx2xml.
%
function stx = xml2stx(xml)
    doc = xmlread(xml);
    root = doc.getDocumentElement();
    attr = root.getAttributes();

    stixels = 0;
    stixelWidth = inf;

    % get some attributes
    for i = 1 : attr.getLength()
        a = attr.item(i-1);
        key = char(a.getName());
        val = char(a.getValue());

        switch key
            case 'numStixels',  stixels = str2num(val);
            case 'stixelWidth', stixelWidth = str2num(val);
        end
    end

    % iterate through the stixel nodes
    nodes = root.getChildNodes();
    n = 0;
 
    for i = 1 : nodes.getLength()
        node = nodes.item(i-1);

        if ~node.hasAttributes, continue; end
        
        attr = node.getAttributes();

        n = n + 1;
        stx(n).width = stixelWidth;

        y0 = 0;
        y1 = 0;

        for j = 1 : attr.getLength()
            aj = attr.item(j-1);
            key = char(aj.getName());
            val = str2num(char(aj.getValue()));

            switch key
                case 'u',    x0 = val - (stixelWidth-1) / 2;
                case 'vT',   y0 = val;
                case 'vB',   y1 = val;
                case 'disp', stx(n).d = val;
            end
        end

        stx(n).x = x0 + 1; % Daimler uses zero-based indexing
        stx(n).y = y0 + 1;
        stx(n).height = y1 - y0 + 1;
        stx(n).dspan = [stx(n).d,stx(n).d];
    end
    
    if n == 0, stx = []; end
end