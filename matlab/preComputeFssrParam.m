% path
%addpath('./smoothpatch_version1b/');

data_path = '../result/result_shapenet/';
save_path = '../result/result_shapenet_ply/';

d = dir([data_path '*.xyz']);

% params
norm_K = 6
nn_K = 3;
%dlist = randperm(numel(d));

parfor ii = 1:numel(d)
    
    iii = ii
    %iii = dlist(ii);
    disp(iii)
    if exist([save_path d(iii).name(1:end-4) '.ply'], 'file')
        continue
    end
    
    pred = load([data_path d(iii).name],'pred');
    pred = [pred(:,1) -pred(:,3) pred(:,2)];

    ptCloud = pointCloud(pred);
    normals = pcnormals(ptCloud,norm_K);

    ptCloud = pointCloud(pred,'normal',normals);
    x = ptCloud.Location(:,1);
    y = ptCloud.Location(:,2);
    z = ptCloud.Location(:,3);
    u = normals(:,1);
    v = normals(:,2);
    w = normals(:,3);

    % calculate scale
    dist = 0;
    Dist = zeros(size(ptCloud.Location,1),1);
    for i = 1:ptCloud.Count
        [~,dists] = findNearestNeighbors(ptCloud,ptCloud.Location(i,:),nn_K);
        dist = dist+mean(dists(2:end));
        Dist(i) = mean(dists(2:end));
    end

    % calculate normal
    ptCloud = pointCloud(pred,'normal',normals);
    
    % save
    propertyNames = {'x','y','z', 'nx', 'ny', 'nz', 'value'};
    propertyValues = {single(x), single(y), single(z), single(u), single(v), single(w), single(Dist)};
    visionPlyWrite([save_path d(iii).name(1:end-4) '.ply'], 'binary_little_endian', 'vertex', propertyNames, propertyValues);
    
end
