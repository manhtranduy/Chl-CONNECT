function [Class, P] = Eumetsat_Class_17(Rrs,opts)
% Syntax:
%   [Class, probability] = Eumetsat_Class_17(Rrs)
%
% Input Arguments:
%   (Required)
%   Rrs                 - Input Remote sensing reflectance
%                           [double | cell]
% Outputs:
%   Class               - 17 OWTs defined from (Melin & Vantrepotte., 2015)
%                           [vector | matrix]
%   P                   - Probability for each Class
%                           [cell]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Manh Tran on 10-07-2023
% Institution: Laboratoire d'Océanologie et de Géosciences - CNRS
% Citation:
% Vantrepotte, V.; Loisel, H.; Dessailly, D.; Mériaux, X. 
% Optical Classification of Contrasted Coastal Waters. 
% Remote Sens. Environ. 2012, 123, 306–323
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    arguments 
        Rrs
        opts.sensor char{mustBeMember(opts.sensor,{'olci','meris','msi','modis'})} = 'olci'
        opts.method char {mustBeMember(opts.method,{'pdf','logreg','svm'})} = 'pdf'
        opts.distribution char {mustBeMember(opts.distribution,{'normal','gamma'})} = 'gamma'
    end

    % Load inputs
    file_path = fileparts(mfilename('fullpath'));
    LUT_dir = fullfile(file_path,'LUTs',upper(opts.sensor),'17OWTs',opts.method);
    if all(ismember(opts.method,'pdf'))
        for i=1:17
            Input_probability.cov_matrix(:,:,i)=readmatrix(fullfile(LUT_dir,sprintf('Cov17_C%i.txt',i)));
            Input_probability.mean_matrix(:,:,i)=readmatrix(fullfile(LUT_dir,sprintf('Mean17_C%i.txt',i)));
        end
    elseif all(ismember(opts.method,'logreg'))
            Input_probability.scaler_mean = readmatrix(fullfile(LUT_dir,'scale_mean.txt'))';
            Input_probability.scaler_std = readmatrix(fullfile(LUT_dir,'scale_std.txt'))';
            Input_probability.logreg_model_coeff = readmatrix(fullfile(LUT_dir,'logreg_model_coeff.txt'))';
            Input_probability.logreg_model_intercept = readmatrix(fullfile(LUT_dir,'logreg_model_intercept.txt'))';
    end

    
    % Load support functions
    HSF=handle_support_functions();
    
    % Sensor Bands
    vis_bands = eval(sprintf('SensorBands.%s_vis',opts.sensor));


    % Handle Rrs input
    if isnumeric(Rrs)
        if size(Rrs,2)~=numel(vis_bands)
            error('Error: Reflectance input must contain %d columns',numel(vis_bands))
        end
    elseif iscell(Rrs)
        if size(Rrs,2)~=numel(vis_bands)
            error('Error: Reflectance input must contain %d cells',numel(vis_bands))
        end
    end

    % Read Rrs input and reshape
    Rrs_input = [];
    for i = 1:numel(vis_bands)
        if iscell(Rrs)
            eval(sprintf('Rrs%d = Rrs{%d};',vis_bands(i),i))
        elseif isnumeric(Rrs)
            eval(sprintf('Rrs%d = Rrs(:,i);',vis_bands(i),i))
        end

        eval(sprintf('[mx,my] = size(Rrs%d);',vis_bands(i)));
        eval(sprintf('Rrs%d = reshape(Rrs%d,[],1);',vis_bands(i),vis_bands(i)))
        eval(sprintf('Rrs_input = [Rrs_input, Rrs%d];',vis_bands(i)));
    end
    

    % Normalize Rrs
    Rrs_norm = normalize_Rrs(Rrs_input,vis_bands);

    % Perform the classification
    [p , Class] = probability(Input_probability,Rrs_norm, ...
                             "method",opts.method, ...
                             "distribution",opts.distribution);


    % Return probability and Class matrices
    for i=1:size(p,2)
        P{i} = reshape(p(:,i),mx,my);
    end

    Class = reshape(Class,mx,my);

end