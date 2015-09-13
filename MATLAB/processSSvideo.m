function [ despreadCube ] = processSSvideo( Video4D, fkernel)
    
    vidLen = size(Video4D, 4);
    kerLen = length(fkernel);

    Grayvideo = zeros(size(Video4D, 1), size(Video4D, 2), kerLen);
    for rep = 0:(floor(vidLen/kerLen) - 1)
        for t = 1:kerLen
            Grayvideo(:,:,t) = Grayvideo(:,:,t) + double(rgb2gray(Video4D(:,:,:,rep*kerLen + t)));
        end
    end
    Grayvideo = Grayvideo / 255;
    
    vidFFT = fft(Grayvideo,kerLen,3);
    despreadFFT = bsxfun(@times, vidFFT, reshape(fkernel, [1 1 kerLen]));
    despreadCube = ifft(despreadFFT,kerLen,3);
    
end