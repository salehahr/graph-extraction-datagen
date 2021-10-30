clear all
close all
clc
%%
% Application() returns:
%      output       A struct that contains the BCOSFIRE filters response
%                   and the final segmented image. In details
%                       - output.respimage: response of the combination of a symmetric and an
%                         asymemtric COSFIRE filters
%                       - output.segmented: the binary output image after
%                         thresholding
%      oriensmap    [optional] map of the orientation that gives the strongest 
%                   response at each pixel. 
%
% If you use this software please cite the following paper:
%
% "George Azzopardi, Nicola Strisciuglio, Mario Vento, Nicolai Petkov, 
% Trainable COSFIRE filters for vessel delineation with application to retinal images, 
% Medical Image Analysis, Volume 19 , Issue 1 , 46 - 57, ISSN 1361-8415

if ~exist('./COSFIRE/dilate')
    BeforeUsing();
end

%% Read Image
VIDEO_NAME = 'GRK021_test';

imageFolder = sprintf('../data/%s/cropped/*', VIDEO_NAME);
ImageFolder_filtered_images = sprintf('../data/%s/filtered/', VIDEO_NAME);

imds          = imageDatastore(imageFolder);
path(path,'./sort_list/');
imds.Files =  natsortfiles(imds.Files);
plotImage = false; 

%% Symmetric filter params
symmfilter = struct();
symmfilter.sigma     = 10;
symmfilter.len       = 10;
symmfilter.sigma0    = 1;
symmfilter.alpha     = 0;

%% Asymmetric filter params
asymmfilter = struct();
asymmfilter.sigma     = 10;
asymmfilter.len       = 20;
asymmfilter.sigma0    = 2;
asymmfilter.alpha     = 0;

preprocess_thresh = 0.1;
%% Filters responses
%size(imds.Files)
disp(['Number of images: ',num2str(size(imds.Files,1))]);
NOImages = size(imds.Files);

for currFrameIdx= 1:NOImages; %204 %1:50
%NOImages = size(imds.Files); 
%for currFrameIdx=1:NOImages(1)
%currFrameIdx = 1;

    [image,fileinfo] = readimage(imds, currFrameIdx);
    image = double(image) ./ 255;
    output = struct();
    tic
    [output.respimage] = BCOSFIRE_media15(image, symmfilter, asymmfilter, preprocess_thresh);
          %  [output.respimage, oriensmap] = BCOSFIRE_media15(image, symmfilter, asymmfilter, 0.6);
        % BCOSFIRE returns:
        %      resp         -> response of the combination of a symmetric and an
        %                   asymemtric COSFIRE filters
        %      oriensmap    -> map of the orientation that gives the strongest 
        %                   response for each pixel. 
        %
        %   The ouput parameter 'oriensmap' is optional. In case it is not
        %   required, the algorithm provides a response image as described in [1].
        %   On the contrary, the response image is computed by first summing up the
        %   symmetric and asymmetric B-COSFIRE filter responses at each
        %   orientation, and then superimposing such responses to achieve rotation
        %   invariance. In this case, the orientation map, with information
        %   about the orientation that gives the strongest response at every pixel,
        %   is provided as output.
    toc
    output.segmented = (output.respimage > 52);
    if plotImage

        figure; imagesc(output.respimage); colormap(gray); axis off; axis image; title('B-COSFIRE response image');

        figure; imagesc(output.segmented); colormap(gray); axis off; axis image; title('B-COSFIRE segmented image');

        figure; imshow(image); title('Original Image')
    end
         %% Save image  1
    myImage =output.respimage./255;   %respimage


      %% Playground -  delete later    

    %     myImage =output.segmented;   %respimage
    %  I=myImage; 
    %  figure(200)
    %  imshow(myImage);
    %  
    %  myImage1 = imgaussfilt(myImage);
    % myImage = imadjust(myImage);
    %  myImage4 = bwmorph(myImage2, 'thin', 1);


     %level = graythresh(myImage2);
     %myImage3 = imbinarize(myImage2,level);
     %Skeleton = bwskel(myImage3,'MinBranchLength', 7);
    %  figure(205)
    %  imshow(myImage4);



    %     myImage1 = adapthisteq(I,'Distribution','exponential','Alpha',0.5,'ClipLimit',0.01);
    %     myImage5 = imsharpen(myImage1,'Radius',1,'Amount',1,'Threshold',0.2);

    %      figure(205)
    %      imshow(myImage5)

    %      figure(201)
    %      imshow(myImage1);
    %   
    %      
    %      myImage2 = imlocalbrighten(myImage,0.4,'alphaBlend',true);
    %      figure(202)
    %      imshow(myImage2);
    %% Save image 2
    [~,name,ext] = fileparts(fileinfo.Filename);
    fullFileName = fullfile(ImageFolder_filtered_images,strcat(name,ext));
    %savename = num2str(sigma);
    %fullFileName = fullfile(ImageFolder_filtered_images,strcat(savename,ext));
    imwrite(myImage, fullFileName);
    %end
    %sigma = sigma + 0.5;
end