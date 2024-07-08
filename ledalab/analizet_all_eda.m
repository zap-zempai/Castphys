%% Folder
folder_name = 'E:\EDA\';

%% Code
% for para todas las carpetas
video_time = tdfread(strcat(folder_name,'video_time.csv'),',');
for i = 1:length(video_time.name)
    % aplicar Ledalab
    v_time = video_time.time(i) + 10;
    n_folder = strcat(strcat(folder_name,video_time.name(i,:)),'\');
    Ledalab(n_folder, 'open', 'text', 'analyze','CDA', 'export_era', [1 v_time .01 1])
end