clc; close all;

record = 'slp01a';
hea_file = [record '.hea'];
dat_file = [record '.dat'];

fid = fopen(hea_file, 'r');
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = lines{1};

parts = strsplit(strtrim(lines{1}));
num_channels = str2double(parts{2});

fs_token = parts{3};
fs = str2double(extractBefore(fs_token, '/'));

fprintf("Record: %s | Channels: %d | Fs: %d Hz\n", record, num_channels, fs);

channel_names = cell(num_channels,1);
gain = zeros(1,num_channels);
baseline = zeros(1,num_channels);

for i = 1:num_channels
    L = strsplit(strtrim(lines{i+1}));

    gain_units = split(L{3}, '/');
    gain(i) = str2double(gain_units{1});

    baseline(i) = str2double(L{6});

    channel_names{i} = strjoin(L(9:end), ' ');
end

disp("Channels:");
disp(channel_names);

fid = fopen(dat_file, 'r');
raw = fread(fid, inf, 'int16');
fclose(fid);

raw = reshape(raw, num_channels, []).';
signals = zeros(size(raw));

for ch = 1:num_channels
    signals(:,ch) = (raw(:,ch) - baseline(ch)) / gain(ch);
end

ecg_idx  = find(contains(channel_names, 'ECG', 'IgnoreCase', true));
resp_idx = find(contains(channel_names, 'Resp', 'IgnoreCase', true));
eeg_idx  = find(contains(channel_names, 'EEG', 'IgnoreCase', true));

ecg_sig  = signals(:, ecg_idx);
resp_sig = signals(:, resp_idx);
eeg_sig  = signals(:, eeg_idx);

% ------------------------------------------------------------
% COMPUTE EDR
% ------------------------------------------------------------
ecg = ecg_sig(:);
N = length(ecg);

ECG = fft(ecg);
f = (0:N-1)*(fs/N);
mask = (f >= 0.5 & f <= 40) | (f >= (fs-40) & f <= (fs-0.5));
ECG_filt = ECG .* mask';
ecg_filt = real(ifft(ECG_filt));

thr = mean(ecg_filt) + 0.5 * std(ecg_filt);
min_dist = round(0.3 * fs);

locs = [];
last_peak = -inf;

for i = 2:length(ecg_filt)-1
    if ecg_filt(i) > thr && ...
       ecg_filt(i) > ecg_filt(i-1) && ...
       ecg_filt(i) > ecg_filt(i+1) && ...
       (i - last_peak) > min_dist
        locs(end+1) = i; %#ok<AGROW>
        last_peak = i;
    end
end

r_amp = ecg_filt(locs);

t_r = locs / fs;
t_full = (1:N) / fs;
edr = interp1(t_r, r_amp, t_full, 'linear', 'extrap');

win = round(0.5 * fs);
edr = movmean(edr, win);
edr = (edr - mean(edr)) / std(edr);

figure; plot(t_full(1:10000), edr(1:10000));
title("ECG-Derived Respiration (EDR) - First 40 seconds");
xlabel("Time (s)"); ylabel("Normalized amplitude");

% ------------------------------------------------------------
% SEGMENTATION
% ------------------------------------------------------------
win_sec = 5;
win = round(win_sec * fs);
hop = round(0.5 * win);
nfft = 256;

segment_sec = 30;
segment_len = segment_sec * fs;
num_segments = floor(length(edr) / segment_len);

out_dir = fullfile('output_images', record);
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fprintf("Generating segmented ECG, EEG, and Respiration images...\n");

target_img_size = [128 128];

for i = 1:num_segments
    idx = (i-1)*segment_len + (1:segment_len);

    % 1. ECG SPECTROGRAM (optional save)
    ecg_seg = ecg_sig(idx);
    [S_ecg, f_ecg, t_ecg] = my_stft(ecg_seg, fs, win, hop, nfft);
    figure('Visible','off');
    imagesc(t_ecg, f_ecg, log1p(S_ecg));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(sprintf('ECG Spectrogram - Segment %d', i));
    colormap(turbo); colorbar;
    saveas(gcf, fullfile(out_dir, sprintf('ecg_%03d.png', i)));
    close;

    % 2. EEG SPECTROGRAM (optional save)
    eeg_seg = eeg_sig(idx);
    [S_eeg, f_eeg, t_eeg] = my_stft(eeg_seg, fs, win, hop, nfft);
    figure('Visible','off');
    imagesc(t_eeg, f_eeg, log1p(S_eeg));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(sprintf('EEG Spectrogram - Segment %d', i));
    colormap(parula); colorbar;
    saveas(gcf, fullfile(out_dir, sprintf('eeg_%03d.png', i)));
    close;

    % 3. RESPIRATION RGB (optional save)
    segment = edr(idx);
    [S, f_r, t_r] = my_stft(segment, fs, win, hop, nfft);
    R = my_mat2gray(log1p(S));

    resp_ds = interp1(linspace(0,1,length(segment)), segment, ...
                      linspace(0,1,size(R,2)), 'linear');
    G = my_cwt_like(resp_ds, fs * size(R,2) / length(segment));
    G = imresize(G, [size(R,1), size(R,2)]);

    env = abs(my_hilbert(segment));
    env_ds = interp1(linspace(0,1,length(env)), env, ...
                     linspace(0,1,size(R,2)), 'linear');
    B = repmat(env_ds(:)', size(R,1), 1);
    B = my_mat2gray(B);

    rgb = cat(3, R, G, B);
    rgb_resized = imresize(rgb, target_img_size);

    figure('Visible','off');
    image(t_r, f_r, rgb_resized);
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(sprintf('Respiration RGB - Segment %d', i));
    saveas(gcf, fullfile(out_dir, sprintf('resp_%03d.png', i)));
    close;
end

% ------------------------------------------------------------
% BUILD JSON GT (SEGMENTED)
% ------------------------------------------------------------
gt.fs = fs;
gt.record = record;
gt.channels = channel_names;
gt.segment_sec = segment_sec;
gt.num_segments = num_segments;

gt.segment_waveforms = cell(num_segments,1);
for i = 1:num_segments
    idx = (i-1)*segment_len + (1:segment_len);
    gt.segment_waveforms{i} = edr(idx)';   % 1 x 7500
end

gt.segment_bpm = zeros(num_segments,1);
for i = 1:num_segments
    idx = (i-1)*segment_len + (1:segment_len);
    seg = edr(idx);

    thr_seg = mean(seg) + 0.3 * std(seg);
    min_dist_seg = round(1.5 * fs);

    locs = [];
    last_peak = -inf;

    for k = 2:length(seg)-1
        if seg(k) > thr_seg && ...
           seg(k) > seg(k-1) && ...
           seg(k) > seg(k+1) && ...
           (k - last_peak) > min_dist_seg
            locs(end+1) = k; %#ok<AGROW>
            last_peak = k;
        end
    end

    gt.segment_bpm(i) = numel(locs) * (60 / segment_sec);
end

fid = fopen(fullfile(out_dir, 'ground_truth_sleep.json'), 'w');
fwrite(fid, jsonencode(gt), 'char');
fclose(fid);

fprintf("Saved JSON + images in %s\n", out_dir);

% ------------------------------------------------------------
% HELPERS
% ------------------------------------------------------------
function [S, f, t] = my_stft(x, fs, win_len, hop, nfft)
    x = x(:);
    win = ones(win_len,1);
    num_frames = floor((length(x)-win_len)/hop)+1;
    S = zeros(nfft/2+1, num_frames);
    for i = 1:num_frames
        idx = (i-1)*hop + (1:win_len);
        frame = x(idx).*win;
        X = fft(frame, nfft);
        S(:,i) = abs(X(1:nfft/2+1));
    end
    f = (0:nfft/2)*fs/nfft;
    t = ((0:num_frames-1)*hop)/fs;
end

function img = my_cwt_like(x, fs)
    x = x(:)';
    scales = 1:64;
    img = zeros(length(scales), length(x));
    for s = scales
        w = sin(2*pi*(1/s)*(1:length(x))/fs);
        img(s,:) = abs(conv(x, w, 'same'));
    end
    img = my_mat2gray(img);
end

function Y = my_mat2gray(X)
    X = double(X);
    mn = min(X(:));
    mx = max(X(:));
    if mx > mn
        Y = (X - mn) / (mx - mn);
    else
        Y = zeros(size(X));
    end
end

function y = my_hilbert(x)
    x = x(:);
    N = length(x);
    X = fft(x);
    h = zeros(N,1);
    if mod(N,2)==0
        h(1) = 1;
        h(N/2+1) = 1;
        h(2:N/2) = 2;
    else
        h(1) = 1;
        h(2:(N+1)/2) = 2;
    end
    y = ifft(X .* h);
end
