%[7 6];%[7 3]; %from http://uk.mathworks.com/help/comm/ref/goldsequencegenerator.html
%   poly2 = [7 3 2 1]

%Videofile = Spreadcode;
%Videofile = insideFlash;
%Videofile = x3OutsideFlash;
%Videofile = farLongerFlash;

[N, codeUSf, fkernel] = gencodes(7, [7 6], [7 3 2 1]);
despreadCube = processSSvideo(Videofile, fkernel);

implay(despreadCube./max(max(max(despreadCube))));

%maxphases = reshape(max(max(despreadCube)), [length(fkernel) 1]);
%[maxPmag, maxPphase] = max(maxphases);
%grayDespread = despreadCube(:,:,maxPphase)./maxPmag;

varmap = var(despreadCube,1,3); %sum of squares also fine, since it is expected to be balanced
[peakval, peakphase] = max(despreadCube, [], 3);
CodeEnergyRatio = peakval.^2 ./ varmap;
phaseOfMaxCER = peakphase(peakval == max(max(peakval)));

%colourDespread = bsxfun(@times, CodeEnergyRatio./max(max(CodeEnergyRatio)), hsv2rgb(phaseOfMaxCER/length(fkernel), 1, 1));
H = peakphase/length(fkernel);
S = ones(size(CodeEnergyRatio, 1), size(CodeEnergyRatio, 2));
V = CodeEnergyRatio./max(max(CodeEnergyRatio));
colourDespread = hsv2rgb(H,S,V);
imshow(colourDespread);