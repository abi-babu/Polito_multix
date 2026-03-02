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
resp_sig = signals(:, resp_idx);   % kept for reference, not used as GT
eeg_sig  = signals(:, eeg_idx);

% ------------------------------------------------------------
% COMPUTE EDR (ECG-DERIVED RESPIRATION) FROM ECG
% ------------------------------------------------------------
ecg = ecg_sig(:);
N = length(ecg);

% 1. FFT-based bandpass filter (0.5–40 Hz)
ECG = fft(ecg);
f = (0:N-1)*(fs/N);
mask = (f >= 0.5 & f <= 40) | (f >= (fs-40) & f <= (fs-0.5));
ECG_filt = ECG .* mask';
ecg_filt = real(ifft(ECG_filt));

% 2. TOOLBOX-FREE R-PEAK DETECTION
thr = mean(ecg_filt) + 0.5 * std(ecg_filt);
min_dist = round(0.3 * fs);

locs = [];
last_peak = -inf;

for i = 2:length(ecg_filt)-1
    if ecg_filt(i) > thr && ...
       ecg_filt(i) > ecg_filt(i-1) && ...
       ecg_filt(i) > ecg_filt(i+1) && ...
       (i - last_peak) > min_dist
       
        locs(end+1) = i;
        last_peak = i;
    end
end

% 3. R-peak amplitudes
r_amp = ecg_filt(locs);

% 4. Interpolate to full sampling rate
t_r = locs / fs;
t_full = (1:N) / fs;
edr = interp1(t_r, r_amp, t_full, 'linear', 'extrap');

% 5. Smooth and normalize
win = round(0.5 * fs);
edr = movmean(edr, win);
edr = (edr - mean(edr)) / std(edr);

% Quick plot
figure; plot(t_full(1:10000), edr(1:10000));
title("ECG-Derived Respiration (EDR) - First 40 seconds");
xlabel("Time (s)"); ylabel("Normalized amplitude");

% ------------------------------------------------------------
% DOWNSTREAM USES EDR AS RESPIRATION SIGNAL
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

fprintf("Generating ECG heatmap...\n");
[S,f,t] = my_stft(ecg_sig, fs, win, hop, nfft);
S_norm = my_mat2gray(S);
ecg_heatmap = uint8(255 * S_norm);
imwrite(ecg_heatmap, fullfile(out_dir, 'ecg_heatmap.png'));

fprintf("Generating EEG spectrogram...\n");
[S,f,t] = my_stft(eeg_sig, fs, win, hop, nfft);
fig = figure('Visible','off');
imagesc(t, f, my_mat2gray(S));
axis xy;
colormap jet;
title('EEG Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
saveas(fig, fullfile(out_dir, 'eeg_spectrogram.png'));
close(fig);

fprintf("Generating segmented Respiration RGB images (using EDR)...\n");

target_img_size = [128 128];

for i = 1:num_segments
    idx = (i-1)*segment_len + (1:segment_len);
    segment = edr(idx);   % use EDR, not Resp(sum)

    % STFT → R channel
    [S,~,~] = my_stft(segment, fs, win, hop, nfft);
    R = my_mat2gray(log1p(S));

    % Wavelet-like → G channel
    resp_ds = interp1(linspace(0,1,length(segment)), segment, ...
                      linspace(0,1,size(R,2)), 'linear');
    G = my_cwt_like(resp_ds, fs * size(R,2) / length(segment));
    G = imresize(G, [size(R,1), size(R,2)]);

    % Envelope → B channel
    env = abs(my_hilbert(segment));
    env_ds = interp1(linspace(0,1,length(env)), env, ...
                     linspace(0,1,size(R,2)), 'linear');
    B = repmat(env_ds(:)', size(R,1), 1);
    B = my_mat2gray(B);

    % Stack into RGB
    rgb = cat(3, R, G, B);

    % FINAL RESIZE (for model input)
    rgb_resized = imresize(rgb, target_img_size);

    % Save
    imwrite(rgb_resized, fullfile(out_dir, sprintf('resp_rgb_%03d.png', i)));
end

gt.fs = fs;
gt.record = record;
gt.channels = channel_names;
gt.segment_sec = segment_sec;
gt.num_segments = num_segments;

% Use EDR as full respiration waveform for Python
gt.respiration_waveform = edr(:)';   % [1 x N], becomes JSON array

disp('--- GT STRUCT ---');
disp(gt);
disp(fieldnames(gt));
json_str = jsonencode(gt);
disp('--- FIRST 300 CHARS OF JSON ---');
disp(json_str(1:300));

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