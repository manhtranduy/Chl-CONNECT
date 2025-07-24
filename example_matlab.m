clear
close all

addpath('./common')

%% Read matchup data
T=readtable('MDB_1990_08_28_2023_07_17_modis_l2gen.csv');


% T = T(~(contains(T.ID, 'SOMLIT') & ...
%       ~ismember(T.Comments, {'2', '6', '7'})),:);
% unique(T.Comments(contains(T.ID, 'SOMLIT')))

T.Comments(ismember(T.Comments, 'Non qualifié'))
T = T(~ismember(T.Comments, 'Non qualifié'), :);


% load Rrs
wl = SensorBands.modis_vis_nir;

for i = 1: length(wl)
    eval(sprintf('Rrs%i_in=T.Rrs_%i;',wl(i),wl(i)))
    eval(sprintf('Rrs%i_l2gen=T.Rrs%i_med;',wl(i),wl(i)))
end

% Construct input
Rrs_input_l2gen=[Rrs412_l2gen Rrs443_l2gen Rrs488_l2gen Rrs531_l2gen Rrs551_l2gen Rrs667_l2gen Rrs748_l2gen];

% Perform CONNECT algorithm
[Chl,Class] = Chl_CONNECT(Rrs_input_l2gen);


% figure('Position', [10 10 700 650]);
% pscatter_update(T.Chla(g),Chl(g),Class(g),...
%     'legendlocation','eastoutside',...
%     'titlefontsize',20,... 
%     'anfontsize',16,...
%     'title','l2gen CONNECT',...
%     'legend','on',...
%     'xlim',[1e-4 1e4],...
%     'ylim',[1e-4 1e4],...
%     'xlabel','Chl-a measured (\mug.L^-^1)',...
%     'ylabel','Chl-a estimated (\mug.L^-^1)',...
%     'transparency',0.5);
% savename='mc_CONNECT_l2gen';
