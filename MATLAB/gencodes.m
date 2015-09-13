function [ N, codeUSf, fkernel ] = gencodes(n, poly1, poly2)

    N = 2^n - 1;
    state1 = ones(n,1);
    state2 = ones(n,1);

    code1 = zeros(N,1);

    for k = 1:N
        newbit = 0;
        for bittap = poly1
            newbit = xor(newbit,state1(bittap));
        end
        code1(k) = newbit;
        state1 = [newbit; state1(1:end-1)];
    end

    codeUSf = filter([1 1],1,upsample(code1,2));
    fkernel = conj(fft(codeUSf));
    fkernel(1) = 0;

end