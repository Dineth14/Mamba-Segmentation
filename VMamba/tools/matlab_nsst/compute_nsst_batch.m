function compute_nsst_batch(image_dir, output_dir)
% compute_nsst_batch  Offline NSST preprocessing using MATLAB NSST toolbox.
%
% Usage:
%   compute_nsst_batch('/path/to/images', '/path/to/output')
%
% This script:
%   - reads RGB images
%   - converts to grayscale
%   - computes NSST coefficients via nsst_dec2
%   - stacks coefficients to (C, H, W)
%   - saves .mat files (coeffs, H, W)

if nargin < 2
    error('Usage: compute_nsst_batch(image_dir, output_dir)');
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

pfilt = 'maxflat';
dfilt = 'dmaxflat7';
levels = [2 2 2];

exts = {'*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp'};
files = [];
for i = 1:numel(exts)
    files = [files; dir(fullfile(image_dir, exts{i}))]; %#ok<AGROW>
end

if isempty(files)
    fprintf('No images found under %s\n', image_dir);
    return;
end

total = numel(files);
fprintf('Found %d images under %s\n', total, image_dir);

for i = 1:total
    fname = files(i).name;
    in_path = fullfile(image_dir, fname);
    [~, stem, ~] = fileparts(fname);

    rgb = imread(in_path);
    if size(rgb, 3) == 3
        gray = rgb2gray(rgb);
    else
        gray = rgb;
    end
    gray = im2double(gray);
    [H, W] = size(gray);

    coeffs = nsst_dec2(gray, levels, pfilt, dfilt);
    coeffs = stack_nsst_coeffs(coeffs);

    if any(isnan(coeffs(:))) || any(isinf(coeffs(:)))
        error('NaN/Inf detected in coefficients for %s', in_path);
    end
    if size(coeffs, 2) ~= H || size(coeffs, 3) ~= W
        error('Shift-invariance check failed for %s', in_path);
    end

    out_path = fullfile(output_dir, [stem, '.mat']);
    save(out_path, 'coeffs', 'H', 'W', '-v7.3');

    if mod(i, 50) == 0 || i == total
        fprintf('Processed %d/%d\n', i, total);
    end
end

fprintf('NSST preprocessing complete: %s\n', output_dir);
end


function coeffs = stack_nsst_coeffs(nsst)
% stack_nsst_coeffs  Convert nsst_dec2 output to (C, H, W).
% nsst is a cell array: {lowpass, dir1, dir2, ...}
%
% Each directional band is itself a cell array.

bands = {};
bands{end+1} = nsst{1}; %#ok<AGROW> low-pass
for s = 2:numel(nsst)
    dirbands = nsst{s};
    for d = 1:numel(dirbands)
        bands{end+1} = dirbands{d}; %#ok<AGROW>
    end
end

C = numel(bands);
[H, W] = size(bands{1});
coeffs = zeros(C, H, W);
for c = 1:C
    coeffs(c, :, :) = bands{c};
end
end
