function [mean_X, std_X, weights_and_biases] = getModelInfo(inputFilePath)



info=ncinfo(inputFilePath);
Layer_Vars={info.Groups.Name};

mean_X=ncread(inputFilePath,'mean_X');
std_X=ncread(inputFilePath,'std_X');
% mean_Y=ncread(inputFilePath,'mean_Y');
% std_Y=ncread(inputFilePath,'std_Y');

for i=1:length(Layer_Vars)
    weights_and_biases{i,1}=ncread(inputFilePath,[Layer_Vars{i} '/weights']);
    weights_and_biases{i,2}=ncread(inputFilePath,[Layer_Vars{i} '/biases']);
end



