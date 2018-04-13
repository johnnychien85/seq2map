function dp = dpinv(dp)
    u = 1 : size(dp,2);
    for i = 1 : size(dp,1)
        [ui,idx] = sort(u-dp(i,:));
        good = find(abs(diff(ui)) > 0);
        dp(i,:) = interp1(ui(good),-dp(i,idx(good)),u);
    end
end