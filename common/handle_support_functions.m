classdef handle_support_functions
    properties
    end

    methods (Static)
        function HSF = handle_support_functions()
        end


        % Handle infinite and imaginary values
        function value = handle_inf_img(value)
            value(imag(value)~=0)=nan;
            value(isinf(value))=nan;
        end

        % Prepare Curve Data
        function [x,y] = prepareCurveXY(x,y)
            x=reshape(x,[],1);y=reshape(y,[],1);
            x=handle_support_functions.handle_inf_img(x);
            y=handle_support_functions.handle_inf_img(y);
            nanRows = any(isnan(x), 2)|any(isnan(y), 2);
            x(nanRows, :) = [];
            y(nanRows, :) = [];
        end

        % Prepare Curve Data
        function varargout = prepareCurve(varargin)
            for i = 1:length(varargin)
                varargin{i}=reshape(varargin{i},[],1);
                if ~islogical(varargin{i})&~isstring(varargin{i})
                    varargin{i}(varargin{i}<=0)=nan;
                    varargin{i}=handle_support_functions.handle_inf_img(varargin{i});
                    varargin{i}(varargin{i}==0)=nan;
                end
            end
            input = table(varargin{:});
            input=rmmissing(input);
            for i = 1:size(input,2)
                varargout{i} = table2array(input(:,i));
            end
        end

        % Get index of brushed data
        function index = get_index_brushedData(target_var,brushedData)
            index= ismember(target_var,brushedData);
        end
    end
end
