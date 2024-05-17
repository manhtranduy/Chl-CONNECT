classdef SensorBands
    properties (Constant)
        % Full bands
        olci =  [400 412 443 490 510 560 620 665 674 681 709 754 761 764 767 779 865];
        meris = [400 412 443 490 510 560 620 665 674 681 709 754 779];
        msi =   [        443 490     560     665         705 740     783];
        modis = [    412 443 488 531 551     667     678     748];
        oli =   [        443 490     560     655                          865];

        % Typical visible bands
        olci_vis =  [412 443 490 510 560 665];
        meris_vis = [412 443 490 510 560 665];
        msi_vis =   [    443 490     560 665];
        modis_vis = [412 443 488 531 551 667];
        oli_vis =   [443     490     560 655];

        % Typical visible & NIR      
        olci_vis_nir =  [412 443 490 510 560 665 709];
        meris_vis_nir = [412 443 490 510 560 665 709];
        msi_vis_nir =   [    443 490     560 665 705];
        modis_vis_nir = [412 443 488 531 551 667     748];
        oli_vis_nir =   [443     490     560 655         865];
    end
    
%     methods (Static)
%         function wavelengths = getWavelengths(sensor)
%             switch lower(sensor)
%                 case 'olci'
%                     wavelengths = SensorWavelengths.olci;
%                 case 'meris'
%                     wavelengths = SensorWavelengths.meris;
%                 case 'msi'
%                     wavelengths = SensorWavelengths.msi;
%                 case 'modis'
%                     wavelengths = SensorWavelengths.modis;
%                 otherwise
%                     error('Unknown sensor');
%             end
%         end
%     end
end