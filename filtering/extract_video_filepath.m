%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\projects\graph-training\config.py
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2021/10/30 19:51:28

%% Initialize variables.
filename = '../config.py';
delimiter = {'= '};
startRow = 3;
endRow = 3;

%% Format for each line of text:
%   column1: text (%s)
%	column2: text (%s)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%s%s%[^''\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
textscan(fileID, '%[^\n\r]', startRow-1, 'WhiteSpace', '', 'ReturnOnError', false);
dataArray = textscan(fileID, formatSpec, endRow-startRow+1, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'TextType', 'string', 'ReturnOnError', false, 'EndOfLine', '\r\n');

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
config = table(dataArray{1:end-1}, 'VariableNames', {'fromfunctions_filesimportmake_folder','filepath'});
VIDEO_FILEPATH_EXT = config.filepath{1}(2:end-1);

%% Clear temporary variables
clearvars filename delimiter startRow endRow formatSpec fileID dataArray ans config;