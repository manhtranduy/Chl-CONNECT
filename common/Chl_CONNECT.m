function [Chl, Class, P, Chla,invalid_mask] = Chl_CONNECT(Rrs,opts)
% Syntax:
%   Chl = Chl_CONNECT(Rrs,varargin)
%
% Input Arguments:
%   (Required)
%   Rrs                 - Input Remote sensing reflectance
%                           [double | cell]
%   (Optional)
%   sensor              - Satellite sensor
%                          'olci' (default)|'meris'|'msi'|'modis'
%                           [char]
%
%   method              - Probability method
%                          'pdf' (default)|'logistic'
%                           [char]
%
%   distribution        - Distribution type (only for Gaussian method)
%                           'normal' (default) | 'gamma'
%                           [char]  
%
%   classmask           - The class desired to mask as nan
%                           [0 (default)| 1| 2| 3| 4| 5...|17]
% Outputs:
%   Chl                 - Chl-a concentration
%                           [vector | matrix]
%   Class               - 5 OWTs defined in (Tran et al., 2023)
%                           [vector | matrix]
%   p                   - Probability for each Class
%                           [vector | matrix]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Manh Tran on 10-07-2023
% updated on: 13-03-2024 for Gloria, LOG, Negri, Valente datasets
% Institution: Laboratoire d'Océanologie et de Géosciences - CNRS
% Citation:
% Tran, M.D.; Vantrepotte, V.; Loisel, H.; Oliveira, E.N.; Tran, K.T.; Jorge, D.; Mériaux, X.; Paranhos, R. 
% Band Ratios Combination for Estimating Chlorophyll-a from Sentinel-2 and Sentinel-3 in Coastal Waters. 
% Remote Sens. 2023, 15, 1653. https://doi.org/10.3390/rs15061653
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    arguments 
        Rrs
        opts.sensor char{mustBeMember(opts.sensor,{'olci','meris','msi','modis'})} = 'modis'
        opts.method char {mustBeMember(opts.method,{'pdf','logreg'})} = 'logreg'
        opts.distribution char {mustBeMember(opts.distribution,{'normal','gamma'})} = 'gamma'
        opts.classmask double {mustBeMember(opts.classmask,0:17)} = 0
        opts.treatRrs logical = false
        opts.logRrsNN logical = false
        opts.logRrsClassif logical = false
        opts.pTransform logical = false
        opts.version double {mustBeMember(opts.version,[5,17])} = 5;
    end

    if strcmp(opts.sensor,'meris')
        opts.sensor='olci';
    end
    % Load inputs
    file_path = fileparts(mfilename('fullpath'));
    LUT_dir = fullfile(file_path,'LUTs', ...
        upper(opts.sensor), ...
        sprintf('%iOWTs',opts.version));
    
    if all(ismember(opts.method,'pdf'))
        OWTs_dir = fullfile(LUT_dir,opts.method);
        for i=1:opts.version
            Input_probability.cov_matrix(:,:,i)=readmatrix(fullfile(OWTs_dir,sprintf('Cov%i_C%i',opts.version,i)));
            Input_probability.mean_matrix(:,:,i)=readmatrix(fullfile(OWTs_dir,sprintf('Mean%i_C%i',opts.version,i)));
        end
    elseif all(ismember(opts.method,'logreg'))
        if opts.logRrsClassif
            OWTs_dir = fullfile(LUT_dir,opts.method,'logRrs');
        else
            OWTs_dir = fullfile(LUT_dir,opts.method,'Rrs');
        end
        Input_probability.scaler_mean = readmatrix(fullfile(OWTs_dir,'scale_mean.txt'))';
        Input_probability.scaler_std = readmatrix(fullfile(OWTs_dir,'scale_std.txt'))';
        Input_probability.logreg_model_coeff = readmatrix(fullfile(OWTs_dir,'logreg_model_coeff.txt'))';
        Input_probability.logreg_model_intercept = readmatrix(fullfile(OWTs_dir,'logreg_model_intercept.txt'))';
    end
    
    
    % Load support functions
    HSF=handle_support_functions();
    
    % Sensor Bands
    switch opts.sensor
        case 'meris'
        case 'olci'
            full_bands = SensorBands.olci;
            vis_bands = SensorBands.olci_vis;
            vis_nir_bands = SensorBands.olci_vis_nir;
        case 'msi'
            full_bands = SensorBands.msi;
            vis_bands = SensorBands.msi_vis;
            vis_nir_bands = SensorBands.msi_vis_nir;
        case 'modis'
            full_bands = SensorBands.modis;
            vis_bands = SensorBands.modis_vis;
            vis_nir_bands = SensorBands.modis_vis_nir;
    end

    
    % Handle Rrs input
    if isnumeric(Rrs)
        if size(Rrs,2)~=numel(vis_nir_bands)
            error('Error: Reflectance input for %s must contain %i columns',opts.sensor,numel(vis_nir_bands))
        end
    elseif iscell(Rrs)
        if size(Rrs,2)~=numel(vis_nir_bands)
            error('Error: Reflectance input for %s must contain %i cells',opts.sensor,numel(vis_nir_bands))
        end
    end

    % Read Rrs input and reshape
    Rrs_input = [];
    for i = 1:numel(vis_nir_bands)
        if iscell(Rrs)
            eval(sprintf('Rrs%i = Rrs{%i};',vis_nir_bands(i),i))
        elseif isnumeric(Rrs)
            eval(sprintf('Rrs%i = Rrs(:,i);',vis_nir_bands(i)))
        end

        eval(sprintf('[mx,my] = size(Rrs%i);',vis_nir_bands(i)));
        eval(sprintf('Rrs%i = reshape(Rrs%i,[],1);',vis_nir_bands(i),vis_nir_bands(i)))
        % if opts.treatRrs
        %     eval(sprintf('Rrs%i(Rrs%i<0) = 10^-4;',vis_nir_bands(i),vis_nir_bands(i)))
        % else
        %     eval(sprintf('Rrs%i(Rrs%i<0) = nan;',vis_nir_bands(i),vis_nir_bands(i)))
        % end
        eval(sprintf('Rrs_input = [Rrs_input, Rrs%i];',vis_nir_bands(i)));
    end
    
    Rrs_classif=Rrs_input(:,1:end-1);
    if opts.treatRrs
        ind=any(Rrs_classif<0,2);
        Rrs_classif(ind,:)=Rrs_classif(ind,:)+abs(min(Rrs_classif(ind,:),[],2))+10^-6;
    end

    % Normalize Rrs
    Rrs_norm = normalize_Rrs(Rrs_classif,vis_bands);

    % Perform the classification
    [p , Class] = probability(Input_probability,Rrs_norm, ...
                             "method",opts.method, ...
                             "distribution",opts.distribution, ...
                             "logRrs",opts.logRrsClassif);
    if opts.pTransform
        p=sqrt(p)./sum(sqrt(p),2); % transformed probability
    end
    Class=reshape(Class,[],1);

    % exclude bad values in NIR band
    % eval(sprintf('p(Rrs%d<=0|isnan(Rrs%d),1)=0;',vis_nir_bands(end),vis_nir_bands(end)))
    eval(sprintf('p(isnan(Rrs%d),1)=0;',vis_nir_bands(end),vis_nir_bands(end)))


    % Return probability and Class matrices
    for i=1:size(p,2)
        P{i} = reshape(p(:,i),mx,my);
    end


    if opts.logRrsNN
        Rrs_input_1=log10(Rrs_input(:,1:end-1));
        Rrs_input_2=log10(Rrs_input);
        inputFilePath=fullfile(LUT_dir,'NN','logRrs');
    else
        Rrs_input_1=Rrs_input(:,1:end-1);
        Rrs_input_2=Rrs_input;
        inputFilePath=fullfile(LUT_dir,'NN','Rrs');
    end

    if strcmp(opts.sensor,'msi')
        Rrs_input_1 = Rrs_input_2;
    end
    
    % Apply clear model
    model_clearPath = fullfile(inputFilePath,'model_clear.h5');
    [mean_X_1, std_X_1, WB_1] = getModelInfo(model_clearPath);
    % [mean_X_2_17, std_X_2_17, WB_2_17] = getModelInfo(model_2_17Path);
    Rrs_input_1 = standardize(Rrs_input_1,mean_X_1,std_X_1);
    Rrs_input_1=HSF.handle_inf_img(Rrs_input_1);

    % Chla{1}=10.^inverse_standardize(predictNN(WB_1,Rrs_input_1),mean_Y_217,std_Y_2_17);
    Chla{1}=10.^predictNN(WB_1,Rrs_input_1);
    Chla{1}=HSF.handle_inf_img(Chla{1});

    % Apply turbid model
    model_turbidPath = fullfile(inputFilePath,'model_turbid.h5');
    [mean_X_2, std_X_2, WB_2] = getModelInfo(model_turbidPath);
    % [mean_X_1, std_X_1, WB_1] = getModelInfo(model_1Path);
    Rrs_input_2 = standardize(Rrs_input_2,mean_X_2,std_X_2);
    Rrs_input_2=HSF.handle_inf_img(Rrs_input_2);

    % Chla{2}=10.^inverse_standardize(predictNN(WB_1,Rrs_input_1),mean_Y_1, std_Y_1);
    Chla{2}=10.^predictNN(WB_2,Rrs_input_2);
    Chla{2}=HSF.handle_inf_img(Chla{2});
    

    if opts.version==5
        p1 = sum(p(:,1:3),2,'omitnan');
        p2 = sum(p(:,4:5),2,'omitnan');
        invalid_mask=~isnan(Chla{1}) & isnan(Chla{2}) & Class~=4 & Class~=5;
        % invalid_mask=~isnan(Chla{1}) & isnan(Chla{2}) & Class~=5;
    else
        p1 = sum(p(:,2:17),2,'omitnan');
        p2 = p(:,1);
        invalid_mask=~isnan(Chla{1}) & isnan(Chla{2}) & Class~=1;
    end
    p2(invalid_mask)=0;
    Chla{2}(invalid_mask)=0;

    % Exclude unrealistic values
    mask1=Chla{1}>15000;
    p1(mask1)=0;
    mask2=Chla{2}>15000;
    p1(mask2)=0;

    % Perform the combination with the OWT-specific probabilities as
    % Blending coefficients
    Chl_n(:,1)=p1.*Chla{1};
    Chl_n(:,2)=p2.*Chla{2};
    Chl=sum(Chl_n,2)./(p1+p2);
    
    % Return Chl
    Class = reshape(Class,mx,my);
    Chl(Class==opts.classmask)=nan;
    Chl=reshape(Chl,mx,my);
end