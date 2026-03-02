function [dff, dff_reg, fmean_raw, fmean_reg, pred_shift, opt_loss_xyz, grid_loss_xyz, grid_pos_xyz] = fcn_Comp3D_main(lib, raw, lib_pos, pixel_list, sin_num)
% lib: H x W x stackNum
% raw: H x W x Frame; data type, double
% lib_pos: stackNum x 1(Z), 3(XYZ)
% pixel_list: 1 x ROI (cell structure)
% sin_num: number of dimensions wants to keep in SVD, integer

if size(lib_pos,2) == 3
    dim = '3d'; % 3D (X,Y,Z) correction
elseif size(lib_pos,2) == 1
    dim = '1d'; % 1D (Z) correction
else
    error('Error. \nInput position dimension must be 1D (z) or 3D (x,y,z) , not %iD.',size(lib_pos,2));
end

addpath('RBF MATLAB');
dist_type = 'cosine'; % distance metric, if not specified, cosine distance
init_num = 10; % number of predicted movement as initial movement 

% Main
% step 1: extracting features using library stacks through SVD dimension reduction
[lib_feature, source_feature, ft_map] = fcn_SVD(double(lib), double(raw), sin_num);
% step 2: finding most similar movement
[pred_shift, opt_loss_xyz, grid_loss_xyz, grid_pos_xyz] = fcn_xyz_interp_rbf(lib_feature, source_feature, lib_pos, dist_type, dim, init_num);
% step 3: compensaion
fmean_raw = (fcn_roi_measure(raw, pixel_list));
fmean_lib = (fcn_roi_measure(lib, pixel_list));
dff = fmean_raw./mean(fmean_raw);
[reg_norm, fmean_raw_pred] = fcn_comp_dff(fmean_lib, fmean_raw, pixel_list, lib_pos, pred_shift);
dff_reg = dff./reg_norm;
fmean_reg = fmean_raw./reg_norm;
end

function [lib_feature, source_feature, V] = fcn_SVD(lib, source, singularNum)
    libSize = size(lib);
    sourceSize = size(source);
    if length(libSize)>2 % Image
        lib = reshape(lib, [], libSize(3)).';          % stackNum x HW
        source = reshape(source, [], sourceSize(3)).'; % Frame x HW
    else % vector
        lib = lib.';
        source = source.';
    end
    % Find library bases that encode displacement info
    [U,S,V] = svds(lib, singularNum);
    % Project both lib and source onto lib bases
    lib_feature = lib * V;          % stackNum x sin_num
    source_feature = source * V;    % Frame x sin_num
    if length(libSize)>2
        V = reshape(V, libSize(1), libSize(2), []); % image bases, H x W x sin_num
    end
end


function [x_opt, opt_loss, grid_loss, x_grid] = fcn_xyz_interp_rbf(lib_feature,source_feature,lib_pos,dist_type,dim,init_num)
sSize = size(source_feature);
if strcmp(dim,'3d') % Comp3D
    x0 = zeros(sSize(1),3);
    x_grid = zeros(sSize(1),3);
    x_opt = zeros(sSize(1),3);
elseif strcmp(dim,'1d') % CompZ
    x0 = zeros(sSize(1),1);
    x_grid = zeros(sSize(1),1);
    x_opt = zeros(sSize(1),1);
end
    
grid_loss = zeros(sSize(1),1); 
opt_loss = zeros(sSize(1), 1);
outside_alarm = false(sSize(1), 1); % true if the predicted movement is beyond max./min. of library position
if ~exist('init_num', 'var')
    init_num = sSize(1); % if init_num not specified, predict based on nearest library position
end
assert(init_num <= sSize(1));

% create interpolation function
F = cell(sSize(2),1);
for a = 1:sSize(2)
    F{a} = rbfcreate(lib_pos', lib_feature(:,a)');
end

% predict closest position by library grid
for m = 1:init_num
    x_grid(m,:) = fcn_init_by_grid(lib_feature, source_feature(m,:), lib_pos);
    grid_loss(m) = cos_loss_fcn(x_grid(m,:), F, source_feature(m,:), sSize);
end

% optimization
options = optimset('MaxFunEvals',1000,'TolX',1e-4,'TolFun',1e-4);
    for j = 1:sSize(1)
        if j <= init_num
            x0(j,:) = x_grid(j,:);
        else
            x0(j,:) = mean(x_opt(j-init_num:j-1,:)); % initial position based on average of previous position
            outside_alarm(j) =  fcn_out_of_range(x_opt(j-1,:),max(lib_pos,[],'all'),min(lib_pos,[],'all'));
        end
        myfun = @(x)cos_loss_fcn(x,F,source_feature(j,:),sSize,dist_type); % calculate similarity
        [x_opt(j,:), opt_loss(j)] = fminsearch(myfun,x0(j,:),options); % predicted position
    if rem(j,1000) == 0; disp(j); end
    end
end

function x2 = cos_loss_fcn(x,F,source_feature,sSize,dist_type)
    if ~exist('dist_type', 'var')
        dist_type = 'cosine';
    end
    v = [];
    for i = 1:sSize(2)
        f = F{i,1}; 
        v = cat(2,v, rbfinterp(x', f));
    end
    x2 = pdist2(v,source_feature,dist_type);
end

function x0 = fcn_init_by_grid(lib_feature,source_feature,lib_pos,dist_type)
    if ~exist('dist_type', 'var')
        dist_type = 'cosine';
    end
    err = zeros(size(lib_feature,1),1);
    for b = 1:size(lib_feature,1)
        err(b) = pdist2(lib_feature(b,:), source_feature, dist_type);
    end
    [~,I] = min(err);
    x0 = lib_pos(I,:);
end

function out_index = fcn_out_of_range(x,pos_max,pos_min)
    out_index = false;
    for i = 1:length(x)
        if (x(i)>pos_max | x(i)<pos_min)
            out_index = true;
        end
    end
end

function [reg_norm, fmean_raw_pred] = fcn_comp_dff(fmean_lib, fmean_raw, pixel_list, lib_pos, pred_shift)
% Calculate baseline fluorescence after motion, and divided by measured fluorescence
% fmean_lib: stackNum x ROI
% fmean_raw: Frame x ROI
F = cell(length(pixel_list),1);
for a = 1:length(pixel_list)
    F{a} = rbfcreate(lib_pos', fmean_lib(:,a)');
end
fmean_raw_pred = zeros(size(fmean_raw));

for j = 1:size(fmean_raw, 1) % frame
    for k = 1:length(pixel_list) % ROI
        f = F{k,1}; 
        fmean_raw_pred(j,k) = rbfinterp(pred_shift(j,:)', f); 
    end
    if rem(j,1000) == 0; disp(j); end
end
reg_norm = fmean_raw_pred./mean(fmean_lib,1);

end

function fmean = fcn_roi_measure(im,pixel_list)
    ss = size(im);
    if length(ss) == 2
        ss(3) = 1;
    end
    im = reshape(im,[],ss(3));
    nroi = length(pixel_list);
    fmean = zeros(ss(3),nroi);
    for i = 1:nroi
        fmean(:,i) = mean(im(pixel_list{i},:))';
    end
end