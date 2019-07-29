clear;clc;close all;
addpath('./smoothpatch_version1b/');
addpath(genpath('./gptoolbox/'));

data_path = '../result/result_shapenet_ply_out/';
save_path = '../result/result_shapenet_ply_out_smooth/';

d = dir([data_path '*-clean.ply']);

patch_size = 600;
pt_num = 2466; %1024 for object-centered evaluation
s_factor = 5;

disp(numel(d))
%ilist = randperm(numel(d));

parfor i = 1:numel(d)

    disp(i)
    %ii = ilist(i);
    ii = i;
    disp(d(ii).name)

    if exist([save_path d(ii).name(1:end-4) '.xyz'], 'file')
        continue
    end

    % read
    [vertex, face] = read_ply([data_path d(ii).name]);
    FV2 = struct('faces', face, 'vertices', vertex);

    % smooth
    FV3 = smoothpatch(FV2,1,s_factor);

    % Poisson-Disc Sampling
    [N,I,B,r] = random_points_on_mesh(FV3.vertices,FV3.faces,pt_num,'Color', 'blue', 'MaxIter', 200);

    % save
    dlmwrite([save_path d(ii).name(1:end-4) '.xyz'],N, 'delimiter',' ');

end
