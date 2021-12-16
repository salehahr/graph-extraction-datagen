clear all
close all
clc

%% get video filepath
extract_video_filepath;

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

%% Settings
plotImage = false;
skip_existing_files = false; 

%% Read Image
% VIDEO_FILEPATH = VIDEO_FILEPATH_EXT(1:end-4);   % read from config.py
VIDEO_FILEPATH = 'S:/data/256/synthetic-bladder11';           % manual entry

imageFolder = sprintf('%s/', VIDEO_FILEPATH);

imds = imageDatastore(imageFolder, 'IncludeSubfolders', true, 'FileExtensions', '.png');
path(path,'./sort_list/');

cropped_matches = strfind(imds.Files, 'cropped');
crop_filter = zeros(length(cropped_matches),1);
for i = 1:length(cropped_matches)
    crop_filter(i) = ~isempty(cropped_matches{i});
end
cropped_imgs = imds.Files(crop_filter>0);   

imds.Files =  natsortfiles(cropped_imgs);

%% Filter params
new_params = [10, 10, 1, 0, 10, 20, 2, 0];      % only coarse structures
old_params = [7, 5, 1, 0, 2.5, 20, 1, 0];       % Studienarbeit Regine
synth_params = [2.3, 3, 1, 0.5, 1, 2, 1.5, 0];  % optimized for synthetic data
params = synth_params;

%% Symmetric filter params
symmfilter = struct();
symmfilter.sigma     = params(1);
symmfilter.len       = params(2);
symmfilter.sigma0    = params(3);
symmfilter.alpha     = params(4);

%% Asymmetric filter params
asymmfilter = struct();
asymmfilter.sigma     = params(5);
asymmfilter.len       = params(6);
asymmfilter.sigma0    = params(7);
asymmfilter.alpha     = params(8);

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
    %% new file name
    [old_folder, name, ext] = fileparts(fileinfo.Filename);
    new_folder = replace(old_folder, 'cropped', 'filtered');
    
    %% make filtered folder if it doesn't already exist
    if ~exist(new_folder, 'dir')
       mkdir(new_folder)
    end
    
    fullFileName = fullfile(new_folder,strcat(name,ext));
    
    if isfile(fullFileName) && skip_existing_files
        continue
    end
    
    %% image
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
    %savename = num2str(sigma);
    %fullFileName = fullfile(ImageFolder_filtered_images,strcat(savename,ext));
    imwrite(myImage, fullFileName);
    %end
    %sigma = sigma + 0.5;
end