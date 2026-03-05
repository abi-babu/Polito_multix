clc; clear; close all;

%% Paths and Setup
rf_base_path = 'F:\bs_breath\rf_path\';
gt_base_path = 'F:\bs_breath\gt\';
output_dir = 'F:\bs_breath\bsbreath_visuals_all';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Sampling Rates
fs_gt = 150;
fs_rf = 200;

%% Process samples one at a time
num_samples = 44;
all_gt_data = cell(1, num_samples);

for sample_idx = 1:num_samples
    fprintf('\n=== Processing sample %d of %d ===\n', sample_idx, num_samples);
    
    % Clear workspace except essential variables
    clearvars -except rf_base_path gt_base_path output_dir fs_gt fs_rf num_samples all_gt_data sample_idx
    
    try
        %% Construct file paths
        rf_path = fullfile(rf_base_path, sprintf('sample_%d.mat', sample_idx));
        gt_path = fullfile(gt_base_path, sprintf('gt_%d.mat', sample_idx));
        
        %% Load GT first (small)
        fprintf('Loading GT file...\n');
        gt_struct = load(gt_path);
        gt_data = double(gt_struct.gt);
        
        %% Load RF data
        fprintf('Loading RF file...\n');
        rf_struct = load(rf_path);
        rf_data = double(rf_struct.rf_channel);
        
        fprintf('RF data dimensions: '); fprintf('%d ', size(rf_data)); fprintf('\n');
        
        %% Normalize
        rf_data = (rf_data - mean(rf_data(:))) / std(rf_data(:));
        gt_data = (gt_data - mean(gt_data(:))) / std(gt_data(:));
        
        %% Store ground truth (small)
        all_gt_data{sample_idx} = gt_data(:)';
        
        %% Export JSON (small text file)
        gt_json_struct = struct();
        gt_json_struct.sample_id = sample_idx;
        gt_json_struct.ground_truth = gt_data(:)';
        
        json_text = jsonencode(gt_json_struct);
        json_path = fullfile(output_dir, sprintf('ground_truth_%d.json', sample_idx));
        fid = fopen(json_path, 'w');
        fwrite(fid, json_text, 'char');
        fclose(fid);
        
        %% Ground Truth Waveform PNG
        t_gt = (0:length(gt_data)-1) / fs_gt;
        fig1 = figure('Visible', 'off');
        plot(t_gt, gt_data, 'LineWidth', 2);
        title(sprintf('Ground Truth Respiration Signal - Sample %d', sample_idx));
        xlabel('Time (s)');
        ylabel('Amplitude');
        grid on;
        saveas(fig1, fullfile(output_dir, sprintf('ground_truth_waveform_%d.png', sample_idx)));
        close(fig1);
        
        %% Clear GT to save memory
        clear gt_data gt_struct
        
        %% RF Heatmap PNG
        fprintf('Generating heatmap...\n');
        if ndims(rf_data) == 3
            rf_heatmap = squeeze(mean(abs(rf_data), 3));
        else
            rf_heatmap = abs(rf_data);
        end
        
        fig2 = figure('Visible', 'off');
        imagesc(rf_heatmap);
        colormap('parula');
        colorbar;
        title(sprintf('RF Heatmap - Sample %d', sample_idx));
        xlabel('Time Index');
        ylabel('Channels');
        saveas(fig2, fullfile(output_dir, sprintf('heatmap_%d.png', sample_idx)));
        close(fig2);
        
        %% RF Spectrogram PNG
        fprintf('Generating spectrogram...\n');
        rf_mean_signal = mean(rf_heatmap, 1);
        
        % Clear heatmap to save memory
        clear rf_heatmap
        
        signal_length = length(rf_mean_signal);
        window = min(128, signal_length);
        noverlap = floor(window/2);
        step = window - noverlap;
        
        w = 0.5 - 0.5*cos(2*pi*(0:window-1)/(window-1));
        num_frames = floor((signal_length - noverlap) / step);
        
        S = zeros(window, num_frames);
        
        for i = 1:num_frames
            idx = (i-1)*step + (1:window);
            frame = rf_mean_signal(idx) .* w;
            S(:, i) = abs(fft(frame));
        end
        
        S = S(1:floor(window/2), :);
        
        fig3 = figure('Visible', 'off');
        imagesc(20*log10(S + eps));
        colormap('jet');
        colorbar;
        title(sprintf('RF Spectrogram - Sample %d', sample_idx));
        xlabel('Time Frames');
        ylabel('Frequency Bins');
        saveas(fig3, fullfile(output_dir, sprintf('spectrogram_%d.png', sample_idx)));
        close(fig3);
        
        % Clear remaining large variables
        clear rf_data rf_mean_signal S
        
        fprintf('Sample %d processed successfully\n', sample_idx);
        
    catch ME
        fprintf('Error processing sample %d: %s\n', sample_idx, ME.message);
    end
    
    % Small pause to let system breathe
    pause(0.5);
end

%% Create single combined JSON with all ground truth signals
fprintf('\nCreating combined ground truth JSON file...\n');

processed = find(~cellfun(@isempty, all_gt_data));
fprintf('Successfully processed %d out of %d samples\n', length(processed), num_samples);

if ~isempty(processed)
    all_gt_json = struct();
    all_gt_json.num_samples = length(processed);
    all_gt_json.fs_gt = fs_gt;
    all_gt_json.processed_samples = processed;
    all_gt_json.samples = cell(1, length(processed));
    
    for idx = 1:length(processed)
        sample_idx = processed(idx);
        all_gt_json.samples{idx} = struct();
        all_gt_json.samples{idx}.sample_id = sample_idx;
        all_gt_json.samples{idx}.ground_truth = all_gt_data{sample_idx};
    end
    
    json_text_all = jsonencode(all_gt_json);
    json_path_all = fullfile(output_dir, 'all_ground_truth.json');
    fid = fopen(json_path_all, 'w');
    fwrite(fid, json_text_all, 'char');
    fclose(fid);
    
    fprintf('Combined JSON saved to: %s\n', json_path_all);
end

%% Final summary
fprintf('\n=== Processing Complete ===\n');
fprintf('Output directory: %s\n', output_dir);
fprintf('Generated files:\n');
fprintf('  - %d heatmap PNGs\n', length(processed));
fprintf('  - %d spectrogram PNGs\n', length(processed));
fprintf('  - %d ground truth waveform PNGs\n', length(processed));
fprintf('  - %d individual JSON files\n', length(processed));
fprintf('  - 1 combined JSON file\n');