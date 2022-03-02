clear
close all
clc

%% Powershell commands
% Expand-Archive .\cropped.zip . -Force
% Write-Host (dir .\**\filtered\*.png | measure).Count;
% ..\7z.exe a -r .\filtered.zip .\**\filtered\*.png

%% get video filepath
% extract_video_filepath;

%%
% If you use this software please cite the following paper:
%
% "George Azzopardi, Nicola Strisciuglio, Mario Vento, Nicolai Petkov, 
% Trainable COSFIRE filters for vessel delineation with application to retinal images, 
% Medical Image Analysis, Volume 19 , Issue 1 , 46 - 57, ISSN 1361-8415

if ~exist('./COSFIRE/dilate', 'file')
    BeforeUsing();
end

%% Settings
plotImage = false;
skip_existing_files = false; 

%% Read Image
% VIDEO_FILEPATH = VIDEO_FILEPATH_EXT(1:end-4); % read from config.py
VIDEO_FILEPATH = 'C:\johann\07_HiWi\Git\SB_20220124_006\cropped'; % manual entry

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
old_params = [7, 5, 1, 0, 2.5, 20, 1, 0];       % Studienarbeit Regine
new_params = [10, 10, 1, 0, 10, 20, 2, 0];      % only coarse structures

synth_params = [2.3, 3, 1, 0.5, 1, 2, 1.5, 0];  % optimized for synthetic data

params_GRK016 = [5, 3, 1, 2, 1, 20, 1, 0];
params_GRK012 = [2.5, 5, 1, 2, 1.5, 10, 1, 0];
params_GRK021 = [3, 5, 1, 0.5, 1.5, 20, 1.5, 0];
params_GRK015 = [2.3, 5, 1, 0.5, 1.5, 13, 0.5, 0];
params_GRK014 = [3, 6, 1, 0.5, 3, 10, 0.5, 0];
params_GRK007 = [3.5, 7, 1, 0.5, 1, 20, 1.5, 0];
params_GRK022 = [4, 6, 1, 2, 0.5, 10, 1, 0];
params_GRK011 = [2, 6, 1, 0, 6, 1, 2, 0];
params_GRK008 = [2, 6, 1, 0, 6, 1, 2, 0];

params = params_GRK014;

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
disp(['Number of images: ',num2str(size(imds.Files,1))]);
NOImages = size(imds.Files);

for currFrameIdx= 1:NOImages
    [image, fileinfo] = readimage(imds, currFrameIdx);
    
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
    [output.respimage] = BCOSFIRE_media15(image, symmfilter, asymmfilter, preprocess_thresh);
    output.segmented = (output.respimage > 52);
    
    if plotImage
        figure; imagesc(output.respimage); colormap(gray); axis off; axis image; title('B-COSFIRE response image');
        figure; imagesc(output.segmented); colormap(gray); axis off; axis image; title('B-COSFIRE segmented image');
        figure; imshow(image); title('Original Image')
    end
    
    %% Save image
    myImage = output.respimage./255;
    imwrite(myImage, fullFileName);
end