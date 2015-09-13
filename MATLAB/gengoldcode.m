function [ fkernel, codeUSf, code] = gengoldcode(n, poly1, poly2, id)
%GENGOLDCODE id should be between 1 and 2^n-1 inclusive

    N = 2^n - 1;
    state1 = ones(n,1);
    state2 = de2bi(id,n)';

    code1 = zeros(N,1);
    code2 = zeros(N,size(id,2));

    for k = 1:N
        newbit1 = 0;
        for bittap = poly1
            newbit1 = xor(newbit1,state1(bittap));
        end
        code1(k) = newbit1;
        state1 = [newbit1; state1(1:end-1)];
        
        newbit2 = zeros(1,size(id,2));
        for bittap = poly2
            newbit2 = xor(newbit2,state2(bittap,:));
        end
        code2(k,:) = newbit2;
        state2 = [newbit2; state2(1:end-1,:)];
    end
    
    code = double(xor(repmat(code1,1,size(id,2)), code2));

    codeUSf = filter([1 1],1,upsample(code,2));
    fkernel = conj(fft(codeUSf));
    fkernel(1,:) = 0;

end