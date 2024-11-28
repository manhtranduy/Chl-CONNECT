function [p, Class, D, Pdf] = probability(Input,Rrs_input, opts)

    % Syntax:
    %   Probability(Input, Rrs_input, varargin)
    %
    % Input Arguments:
    %   (Required)
    %   Input               - Input corresponding to the probability method
    %                         {cov_matrix, mean_matrix} - 'gaussian'
    %                         coefficients - 'logistic'
    %
    %   Rrs_input            - Normalized remote sensing reflectance
    %                           [vector | matrix]
    %   (Optional)
    %   method              - Probability method
    %                          'pdf' (default)|'logistic'
    %                           [char]
    %
    %   distribution        - Distribution type (only for Gaussian method)
    %                           'normal' (default) | 'gamma'
    %                           [char]   
    % Outputs:
    %   p                   - Probability of each defined Class
    %                           [vector | matrix]  
    % 
    %   Class               - Class retrieval according to maximum probability
    %                           [vector | matrix]    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Default arguments
    arguments
        Input
        Rrs_input
        opts.method char {mustBeMember(opts.method,{'pdf','logreg','tree','svm','naivebayes','adaboostm2'})} = 'pdf'
        opts.distribution char {mustBeMember(opts.distribution,{'normal','gamma'})} = 'normal'
        opts.logRrs logical = true
    end
    
    % Config Inputs
    if strcmp(opts.method,'pdf')
        if ~iscell(Input)&&numel(Input)~=2
            cov_matrix=Input.cov_matrix;
            mean_matrix=Input.mean_matrix;
        else
            cov_matrix=Input{1};
            mean_matrix=Input{2};
        end

        % Numer of Class
        nc=size(mean_matrix,3);
        % Number of band involved
        b=size(mean_matrix,2);
    end

    % Load support functions
    HSF= handle_support_functions();

    % Handle Input Rrs
    if opts.logRrs
        Rrs_input=log10(double(Rrs_input));
    end
    Rrs_input=HSF.handle_inf_img(Rrs_input);
    
   
    %% Get Maximum likely hood

    switch opts.method
        case 'pdf'
            D = nan(size(Rrs_input,1),nc);
            Pdf = nan(size(Rrs_input,1),nc);
            for i=1:nc
                D(:,i)=(pdist2(Rrs_input,mean_matrix(:,:,i),'mahalanobis',cov_matrix(:,:,i))).^2;
                % D(:,i)=(pdist2(Rrs_input,mean_matrix(:,:,i),'mahalanobis',cov_matrix(:,:,i)));
                % distribution
                switch opts.distribution
                    case 'normal'
                        threshold_D=1488;
                        Pdf(:,i)=mvnpdf(Rrs_input,mean_matrix(:,:,i),cov_matrix(:,:,i));
                        % MS=((2*pi)^(b/2))*(det(cov_matrix(:,:,i)).^(1/2));
                        % Pdf(:,i)=(1/MS)*(exp(-0.5.*D(:,i)));
                    case 'gamma'
                        threshold_D=432;
                        Pdf(:,i)=gammainc(b/2,D(:,i)./2,'lower')./gamma(b/2);
                        % Pdf(:,i)=gammainc(b/2,sqrt(D(:,i))./2,'lower')./gamma(b/2);
                end
    
            end
            unclassified_ind = all(Pdf==0,2);
            D_tmp=D(unclassified_ind,:);
            for k=1:size(D_tmp,1)
                while any(D_tmp(k,:)>threshold_D)
                    D_tmp(k,:)=sqrt(D_tmp(k,:));
                end
            end

            D(unclassified_ind,:)=D_tmp;
            for i=1:nc
                % distribution
                switch opts.distribution
                    case 'normal'
                        % Pdf(unclassified_ind,i)=mvnpdf(Rrs_input,mean_matrix(:,:,i),cov_matrix(:,:,i));
                        MS=((2*pi)^(b/2))*(det(cov_matrix(:,:,i)).^(1/2));
                        Pdf(unclassified_ind,i)=(1/MS)*(exp(-0.5.*D(unclassified_ind,i)));
                    case 'gamma'
                        Pdf(unclassified_ind,i)=gammainc(b/2,D(unclassified_ind,i)./2,'lower')./gamma(b/2);
                end
    
            end

            
        
            % Calculate the normalized probability from Pdf
            Pdf=double(Pdf);
            p=Pdf./sum(Pdf,2);

        case 'logreg'
            % try
            %     p = mnrval(Input,Rrs_input);
            % catch
            %     [~,p] = Input.predict(Rrs_input);
            %     p=sqrt(p)./sum(sqrt(p),2); % transformed probability
            % end
            Rrs_input_scale = (Rrs_input - Input.scaler_mean)./ Input.scaler_std;
            log_odds = Rrs_input_scale* Input.logreg_model_coeff + Input.logreg_model_intercept;
            max_log_odds = max(log_odds, [], 2);

            p = exp(log_odds-max_log_odds) ./ sum(exp(log_odds-max_log_odds), 2);

        case 'tree'
            [~,p]=predict(Input, Rrs_input);

        case 'svm'
            [~, scores] = predict(Input,Rrs_input);
            p = 10.^(scores) ./ sum(10.^(scores), 2); 

        case 'naivebayes'
            [~,p]=predict(Input, Rrs_input);

        case 'adaboostm2'
            [~, scores] = predict(Input,Rrs_input);
            p = scores ./ sum(scores, 2);
            
    end
    

    [Val,Class]=max(p,[],2);


    % Class(Val==0|isnan(Val))=nan;
    Class(isnan(Val))=nan;
    Class(any(isnan(Rrs_input),2))=nan;
    for i=1:size(p,2)
        p(isnan(Class))=nan;
    end
    % Class=int8(Class);
end




