function [dff, dff_reg, fmean_raw, fmean_reg, pred_shift, opt_loss_xyz, grid_loss_xyz, grid_pos_xyz] = fcn_Comp3D_main(ref, raw, ref_pos, pixel_list, sin_num)
% ref: M x N x Stack
% raw: M x N x Frame, need to be double
% ref_pos: stack x 1(Z), 3(XYZ)
% pixel_list: 1 x ROI (cell structure)
% sin_num: integer, how many dimension kept

if size(ref_pos,2) == 3
    dim = '3d'; % 3D (X,Y,Z) correction
elseif size(ref_pos,2) == 1
    dim = '1d'; % 1D (Z) correction
else
    error('Error. \nInput position dimension must be 1D (z) or 3D (x,y,z) , not %iD.',size(ref_pos,2));
end
if ~exist('dist_type', 'var')
    dist_type = 'cosine'; % distance metric, if not specified, cosine distance
end
[ref_feature, source_feature, ft_map] = fcn_SVD(double(ref), double(raw), sin_num);
init_num = 10;
[pred_shift, opt_loss_xyz, grid_loss_xyz, grid_pos_xyz] = fcn_xyz_interp_rbf(ref_feature,source_feature,ref_pos,dist_type,dim,init_num);

fmean_raw = (fcn_roi_measure(raw, pixel_list));
fmean_ref = (fcn_roi_measure(ref, pixel_list));
dff = fmean_raw./mean(fmean_raw);

[reg_norm, fmean_raw_pred] = fcn_comp_dff(fmean_ref, fmean_raw, pixel_list, ref_pos, pred_shift);
dff_reg = dff./reg_norm;
fmean_reg = fmean_raw./reg_norm;

end


function [ref_feature, source_feature, V] = fcn_SVD(lib, source, singularNum)
    % Library size: M * N * Position
    % Source size: M * N * Frame
    % SingularNum: integer
    libSize = size(lib);
    sourceSize = size(source);
    if length(libSize)>2
        lib = reshape(lib, [], libSize(3)).';          % ref size: Position * MN
        source = reshape(source, [], sourceSize(3)).'; % source size: Frame * MN
    else
        lib = lib.';
        source = source.';
    end
    % Find library bases that encode movement info
    [U,S,V] = svds(lib, singularNum);
    % Project both lib and source onto lib bases
    ref_feature = lib * V;          % lib_feature size: Position * singularNum
    source_feature = source * V;    % source_feature size: Frame * singularNum
    if length(libSize)>2
        V = reshape(V, libSize(1), libSize(2), []); % Bases size: M x N x singularNum
    end
end


function [x_opt, opt_loss, grid_loss, x_grid] = fcn_xyz_interp_rbf(lib_feature,source_feature,ref_pos,dist_type,dim,init_num)
% addpath('\\120.126.51.3\data\John\code\AddOn\RBF MATLAB');

% lib_feature: Position x singularNum
% source feature: Frame x singularNum
% ref_pos: Position x 3
% init_num: The number of previous predicted movement as initial movement 
sSize = size(source_feature);
if strcmp(dim,'3d')
    x0 = zeros(sSize(1),3);
    x_grid = zeros(sSize(1),3);
    x_opt = zeros(sSize(1),3);
elseif strcmp(dim,'1d')
    x0 = zeros(sSize(1),1);
    x_grid = zeros(sSize(1),1);
    x_opt = zeros(sSize(1),1);
end
    
grid_loss = zeros(sSize(1),1);
opt_loss = zeros(sSize(1), 1);
outside_alarm = false(sSize(1), 1);
comp_vec = zeros(sSize(1),sSize(2));
if ~exist('init_num', 'var')
    init_num = sSize(1); % if not specified, predict based on nearest library position
end
assert(init_num <= sSize(1));

% create interpolation function
F = cell(sSize(2),1);
for a = 1:sSize(2)
    F{a} = rbfcreate(ref_pos', lib_feature(:,a)');
end

% Closest position predicted by library grid
for m = 1:init_num
    x_grid(m,:) = fcn_init_by_grid(lib_feature, source_feature(m,:), ref_pos);
    grid_loss(m) = cos_loss_fcn(x_grid(m,:), F, source_feature(m,:), sSize);
end

% optimization
% options = optimset('MaxFunEvals',1000,'TolX',2e-6,'TolFun',2e-6);
options = optimset('MaxFunEvals',1000,'TolX',1e-4,'TolFun',1e-4);
    for j = 1:sSize(1)
        if j <= init_num
            x0(j,:) = x_grid(j,:);
        else
            x0(j,:) = mean(x_opt(j-init_num:j-1,:)); % initial position based on average of previous position
            outside_alarm(j) =  fcn_out_of_range(x_opt(j-1,:),max(ref_pos,[],'all'),min(ref_pos,[],'all'));
        end
        myfun = @(x)cos_loss_fcn(x,F,source_feature(j,:),sSize,dist_type); % calculate similarity
        [x_opt(j,:), opt_loss(j)] = fminsearch(myfun,x0(j,:),options); % predicted position
%         for k = 1:sSize(2)
%             f = F{k,1}; 
%             comp_vec(j,k) = rbfinterp([0,0,0]',f) - rbfinterp(x_opt(j,:)',f); % compensated features
%         end
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

function x0 = fcn_init_by_grid(ref_feature,source_feature,ref_pos,dist_type)
    if ~exist('dist_type', 'var')
        dist_type = 'cosine';
    end
    err = zeros(size(ref_feature,1),1);
    for b = 1:size(ref_feature,1)
        err(b) = pdist2(ref_feature(b,:), source_feature, dist_type);
    end
    [~,I] = min(err);
    x0 = ref_pos(I,:);
end

function out_index = fcn_out_of_range(x,pos_max,pos_min)
    out_index = false;
    for i = 1:length(x)
        if (x(i)>pos_max | x(i)<pos_min)
            out_index = true;
        end
    end
end

function [reg_norm, fmean_raw_pred] = fcn_comp_dff(fmean_ref, fmean_raw, pixel_list, ref_pos, pred_shift)
% Calculate baseline fluorescence after motion, and divided by measured fluorescence
% fmean_ref: Position x ROI
% fmean_raw: Frame x ROI
F = cell(length(pixel_list),1);
for a = 1:length(pixel_list)
    F{a} = rbfcreate(ref_pos', fmean_ref(:,a)');
end
fmean_raw_pred = zeros(size(fmean_raw));

for j = 1:size(fmean_raw, 1) % frame
    for k = 1:length(pixel_list) % ROI#
        f = F{k,1}; 
        fmean_raw_pred(j,k) = rbfinterp(pred_shift(j,:)', f); 
    end
    if rem(j,1000) == 0; disp(j); end
end
reg_norm = fmean_raw_pred./mean(fmean_ref,1);

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