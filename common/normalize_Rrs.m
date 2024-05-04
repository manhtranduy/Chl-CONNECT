function Rrs_norm = normalize_Rrs(Rrs,wave_lengths)

%Normalize the reflectance 
    area=trapz(wave_lengths,Rrs,2);
    Rrs_norm=Rrs./area;
end