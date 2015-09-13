%[7 6];%[7 3]; %from http://uk.mathworks.com/help/comm/ref/goldsequencegenerator.html
%   poly2 = [7 3 2 1]

%Videofile = Spreadcode;
%Videofile = insideFlash;
%Videofile = x3OutsideFlash;
Videofile = farLongerFlash;
%Videofile = glassFlash;
%Videofile = x1OutsideFlash(:,:,:,1:1000);

[N, codeUSf, fkernel] = gencodes(7, [7 6], [7 3 2 1]);
despreadCube = processSSvideo(Videofile, fkernel);

%implay(despreadCube./max(max(max(despreadCube))));

varmap = var(despreadCube,1,3); %sum of squares also fine, since it is expected to be balanced
[peakval, peakphase] = max(despreadCube, [], 3);
localCodeEnergyRatio = peakval.^2 ./ varmap;

phaseOfMaxCER = peakphase(localCodeEnergyRatio == max(max(localCodeEnergyRatio)));
CERofMaxPhase = despreadCube(:,:,phaseOfMaxCER).^2 ./ varmap;

%colourDespread = bsxfun(@times, CodeEnergyRatio./max(max(CodeEnergyRatio)), hsv2rgb(phaseOfMaxCER/length(fkernel), 1, 1));
H = peakphase/length(fkernel);
S = ones(size(localCodeEnergyRatio, 1), size(localCodeEnergyRatio, 2));
V = localCodeEnergyRatio./max(max(localCodeEnergyRatio));
colourDespread = hsv2rgb(H,S,V);


%figure(1);
subtightplot(2,2,1)
imshow(CERofMaxPhase./max(max(CERofMaxPhase)));

%figure(2);
subtightplot(2,2,2)
imshow(colourDespread);

%figure(3);
subtightplot(2,2,3)
imshow(Videofile(:,:,:,1));

%figure(4);
subtightplot(2,2,4)
imshow(despreadCube(:,:,phaseOfMaxCER)./max(max(despreadCube(:,:,phaseOfMaxCER))));

