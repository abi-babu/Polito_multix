clc; clear; close all;

%% Paths and Setup
rf_path = 'rf_data/rf_channel.mat';
gt_path = 'gt_data/gt_1.mat';
output_dir = 'bsbreath_visuals';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Sampling Rates
fs_gt = 150;  % Ground truth respiration signal
fs_rf = 200;  % RF signal

%% Load Data
rf_struct = load(rf_path);
gt_struct = load(gt_path);

rf_data = double(rf_struct.rf_channel);  % [channels × time] or [antennas × subcarriers × time]
gt_data = double(gt_struct.gt);          % [1 × time] or [time × 1]

%% Normalize
rf_data = (rf_data - mean(rf_data(:))) / std(rf_data(:));
gt_data = (gt_data - mean(gt_data(:))) / std(gt_data(:));

%% 1. Ground Truth Respiration Waveform
t_gt = (0:length(gt_data)-1) / fs_gt;
fig1 = figure('Visible','on');
plot(t_gt, gt_data, 'LineWidth', 2);
title('Ground Truth Respiration Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
set(fig1, 'Position', [100, 100, 800, 300]);
saveas(fig1, fullfile(output_dir, 'ground_truth.png'));

%% 2. RF Heatmap (Ensure 2D and Real)
if ndims(rf_data) == 3
    rf_heatmap = squeeze(mean(rf_data, 3));  % [channels × subcarriers]
else
    rf_heatmap = rf_data;  % Already 2D
end

fig2 = figure('Visible','on');
imagesc(abs(rf_heatmap));  % Convert complex to magnitude
colormap('parula');        
colorbar;
title('RF Heatmap (Channels × Time)');
xlabel('Time Index');
ylabel('Channels');
set(fig2, 'Position', [100, 100, 800, 500]);
saveas(fig2, fullfile(output_dir, 'rf_heatmap.png'));

%% 3. RF Spectrogram (Mean across channels)
rf_mean = mean(abs(rf_heatmap), 1);  % Collapse to 1D and ensure real

signal_length = length(rf_mean);
window = min(128, signal_length);       
noverlap = floor(window / 2);
nfft = max(256, 2^nextpow2(window));

fig3 = figure('Visible','on');
spectrogram(rf_mean, window, noverlap, nfft, fs_rf, 'yaxis');
title('RF Spectrogram');
set(fig3, 'Position', [100, 100, 800, 400]);
saveas(fig3, fullfile(output_dir, 'rf_spectrogram.png'));
