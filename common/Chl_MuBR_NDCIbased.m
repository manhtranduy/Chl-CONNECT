function [Chl, Class, P, Chla] = Chl_MuBR_NDCIbased(Rrs,opts)
% Syntax:
%   Chl = Chl_MuBR_NDCIbased(Rrs,varargin)
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
%                          'gaussian' (default)|'logistic'
%                           [char]
%
%   distribution        - Distribution type (only for Gaussian method)
%                           'normal' (default) | 'gamma'
%                           [char]  
%
%   classmask           - The class desired to mask as nan
%                           [0| 1| 2| 3| 4| 5 (default)]
% Outputs:
%   Chl                 - Concentration of Chlorophyll-a estimated using
%   MuBR model for OWTs 1, 2, and 3 and NDCI-based model for OWT 4
%                           [vector | matrix]
%   Class               - 5 OWTs defined from (Tran et al., 2023)
%                           [vector | matrix]
%   p                   - Probability for each Class
%                           [vector | matrix]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Manh Tran on 10-07-2023
% updated on: 16-07-2023 for Gloria, LOG, Negri datasets
% Institution: Laboratoire d'Océanologie et de Géosciences - CNRS
% Citation:
% Tran, M.D.; Vantrepotte, V.; Loisel, H.; Oliveira, E.N.; Tran, K.T.; Jorge, D.; Mériaux, X.; Paranhos, R. 
% Band Ratios Combination for Estimating Chlorophyll-a from Sentinel-2 and Sentinel-3 in Coastal Waters. 
% Remote Sens. 2023, 15, 1653. https:\doi.org/10.3390/rs15061653
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    arguments 
        Rrs
        opts.sensor char{mustBeMember(opts.sensor,{'olci','meris','msi','modis','oli'})} = 'olci'
        opts.method char {mustBeMember(opts.method,{'pdf','logreg','svm'})} = 'pdf'
        opts.distribution char {mustBeMember(opts.distribution,{'normal','gamma'})} = 'normal'
        opts.classmask double {mustBeMember(opts.classmask,[0,1,2,3,4,5])} = 0
        opts.version int8 {mustBeMember(opts.version,[1,2])} = 1 %
        opts.spectralShift logical = false
        opts.pTransform logical = false
        opts.logRrsClassif logical = false
    end

    if strcmp(opts.sensor,'meris')
        opts.sensor='olci';
    end

    % Load inputs
    file_path = fileparts(mfilename('fullpath'));
    LUT_dir = fullfile(file_path,'LUTs', ...
        upper(opts.sensor), ...
        '5OWTs');
    
    if all(ismember(opts.method,'pdf'))
        OWTs_dir = fullfile(LUT_dir,opts.method);
        for i=1:5
            Input_probability.cov_matrix(:,:,i)=readmatrix(fullfile(OWTs_dir,sprintf('Cov5_C%i',i)));
            Input_probability.mean_matrix(:,:,i)=readmatrix(fullfile(OWTs_dir,sprintf('Mean5_C%i',i)));
        end
        opts.logRrsClassif=true;
    elseif all(ismember(opts.method,'logreg'))
        OWTs_dir = fullfile(LUT_dir,opts.method,'Rrs');
        Input_probability.scaler_mean = readmatrix(fullfile(OWTs_dir,'scale_mean.txt'))';
        Input_probability.scaler_std = readmatrix(fullfile(OWTs_dir,'scale_std.txt'))';
        Input_probability.logreg_model_coeff = readmatrix(fullfile(OWTs_dir,'logreg_model_coeff.txt'))';
        Input_probability.logreg_model_intercept = readmatrix(fullfile(OWTs_dir,'logreg_model_intercept.txt'))';
        opts.logRrsClassif=false;
    end

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
        case 'oli'
            full_bands = SensorBands.oli;
            vis_bands = SensorBands.oli_vis;
            vis_nir_bands = SensorBands.oli_vis_nir;
    end

    % Handle Rrs input
    if isnumeric(Rrs)
        if size(Rrs,2)~=numel(vis_nir_bands)
            error('Error: Reflectance input for %s must contain %d columns',opts.sensor,numel(vis_nir_bands))
        end
    elseif iscell(Rrs)
        if size(Rrs,2)~=numel(vis_nir_bands)
            error('Error: Reflectance input for %s must contain %d cells',opts.sensor,numel(vis_nir_bands))
        end
    end

    % Read Rrs input and reshape
    Rrs_input = [];
    for i = 1:numel(vis_nir_bands)
        if iscell(Rrs)
            eval(sprintf('Rrs%d = Rrs{%d};',vis_nir_bands(i),i))
        elseif isnumeric(Rrs)
            eval(sprintf('Rrs%d = Rrs(:,i);',vis_nir_bands(i),i))
        end

        eval(sprintf('[mx,my] = size(Rrs%d);',vis_nir_bands(i)));
        eval(sprintf('Rrs%d = reshape(Rrs%d,[],1);',vis_nir_bands(i),vis_nir_bands(i)))
        eval(sprintf('Rrs_input = [Rrs_input, Rrs%d];',vis_nir_bands(i)));
    end

    Rrs_classif=Rrs_input(:,1:end-1);
    if opts.spectralShift
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
    eval(sprintf('p(Rrs%d<=0|isnan(Rrs%d),4:5)=0;',vis_nir_bands(end),vis_nir_bands(end)))

    % Return probability and Class matrices
    for i=1:size(p,2)
        P{i} = reshape(p(:,i),mx,my);
    end

    % Define Band Ratios according to sensor
    switch opts.sensor
        case 'meris'
        case 'olci'
            eval(sprintf('R1=Rrs%d./Rrs%d;',full_bands(4),full_bands(3)));
            eval(sprintf('R2=Rrs%d./Rrs%d;',full_bands(6),full_bands(4)));
            eval(sprintf('R3=Rrs%d./Rrs%d;',full_bands(8),full_bands(6)));
            eval(sprintf('R=(Rrs%d-Rrs%d)./(Rrs%d+Rrs%d);',full_bands(11),full_bands(8),full_bands(11),full_bands(8)));
        case 'msi'
            eval(sprintf('R1=Rrs%d./Rrs%d;',full_bands(2),full_bands(1)));
            eval(sprintf('R2=Rrs%d./Rrs%d;',full_bands(3),full_bands(2)));
            eval(sprintf('R3=Rrs%d./Rrs%d;',full_bands(4),full_bands(3)));
            eval(sprintf('R=(Rrs%d-Rrs%d)./(Rrs%d+Rrs%d);',full_bands(5),full_bands(4),full_bands(5),full_bands(4)));
        case 'modis'
            eval(sprintf('R1=Rrs%d./Rrs%d;',full_bands(3),full_bands(2)));
            eval(sprintf('R2=Rrs%d./Rrs%d;',full_bands(5),full_bands(3)));
            eval(sprintf('R3=Rrs%d./Rrs%d;',full_bands(6),full_bands(5)));
            eval(sprintf('R=(Rrs%d-Rrs%d)./(Rrs%d+Rrs%d);',full_bands(8),full_bands(6),full_bands(8),full_bands(6)));

        case 'oli'
        eval(sprintf('R1=Rrs%d./Rrs%d;',full_bands(2),full_bands(1)));
        eval(sprintf('R2=Rrs%d./Rrs%d;',full_bands(3),full_bands(2)));
        eval(sprintf('R3=Rrs%d./Rrs%d;',full_bands(4),full_bands(3)));
        eval(sprintf('R4=Rrs%d./Rrs%d;',full_bands(5),full_bands(4)));
    end

    % Apply MuBR model for OWTs 1, 2, and 3
    coef123 = readmatrix(fullfile(LUT_dir,'MuBR_NDCI',sprintf('coef123_v%i.txt',opts.version)));
    coef4 = readmatrix(fullfile(LUT_dir,'MuBR_NDCI',sprintf('coef4_v%i.txt',opts.version)));

    Chla{1}=10.^(coef123(1)+coef123(2).*log10(R1)+coef123(3).*log10(R2)+coef123(4).*log10(R3));
    Chla{1}=HSF.handle_inf_img(Chla{1});
    
    % Apply NDCI-based model for OWT 4
    if strcmp(opts.sensor,'oli')
        Chla{2}=10.^(coef4(1) + coef4(2).*log10(R1)+ coef4(3).*log10(R2)+ ...
                                coef4(4).*log10(R3)+ coef4(5).*log10(R4));
    else
        Chla{2}=10.^(coef4(1) + coef4(2).*R +coef4(3).*R.^2);
    end
    Chla{2}=HSF.handle_inf_img(Chla{2});
    % exclude bad values in NIR band
    eval(sprintf('Chla{2}(Rrs%i<=0|isnan(Rrs%i)&Class~=4&~isnan(Class))=0;',vis_nir_bands(end),vis_nir_bands(end)))

    % Perform the combination with the OWT-specific probabilities as
    % Blending coefficients
    if opts.classmask==5
        Chl_n(:,1)=sum(p(:,1:3),2).*Chla{1};
        Chl_n(:,2)=sum(p(:,4),2).*Chla{2};
        Chl=sum(Chl_n,2)./sum(p(:,1:4),2);
    elseif opts.classmask<5
        Chl_n(:,1)=sum(p(:,1:3),2).*Chla{1};
        Chl_n(:,2)=sum(p(:,4:5),2).*Chla{2};
        Chl=sum(Chl_n,2)./sum(p(:,1:5),2);
    else
        error('Error: Class number must be lower than 5');
    end

    % Return Chl
    Chl(Class==opts.classmask)=nan;
    Chl=reshape(Chl,mx,my);
    Class = reshape(Class,mx,my);
end