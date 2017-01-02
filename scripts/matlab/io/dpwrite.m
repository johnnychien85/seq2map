function dpwrite(dp,filename,dmax)
    dp = uint16(dp / dmax * 65535);
    imwrite(dp,filename);
end