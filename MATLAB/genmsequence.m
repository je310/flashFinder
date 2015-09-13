function [ fkernel, codeUSf, code] = genmsequence(n, poly)

    N = 2^n - 1;
    state = ones(n,1);
    code = zeros(N,1);

    for k = 1:N
        newbit = 0;
        for bittap = poly
            newbit = xor(newbit,state(bittap));
        end
        code(k) = newbit;
        state = [newbit; state(1:end-1)];
    end

    codeUSf = filter([1 1],1,upsample(code,2));
    fkernel = conj(fft(codeUSf));
    fkernel(1) = 0;

end