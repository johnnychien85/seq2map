function x = eucl2cano(x,K)
    x = homo2eucl(eucl2homo(x)*inv(K)');
end