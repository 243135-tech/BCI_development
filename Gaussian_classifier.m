%% Optimized Data Processing for Motor Imagery BCI
% Luca Boggiani - Optimized for Multiple File Processing
clc; clear; close all;

%% Parameters
fs = 512; % Sampling Rate
T = 1/fs;
ntrial = 20;
nchannel = 15;
wlength = 0.5; % Spectrogram window length in seconds
pshift = 0.25; % Shift of the internal PSD windows
wshift = 0.0625; % External window shift in seconds
mlength = 1; % Moving average window length

% Load all .mat files in the directory
data_files = dir('Trainings/*.mat'); 
file_count = length(data_files);

% EEG Channel Labels
channel_labels = {'F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', ...
                  'FC3', 'FCz', 'FC4', 'CP3', 'CPz', 'CP4'};

%% Preprocessing: Loop Through All Files
all_trials = {}; % Store extracted trials
all_labels = {}; % Store corresponding labels
all_psd_features = {}; % Store extracted PSD features
all_baseline_features = {}; % Store baseline PSD features
all_frequencies = {}; % Store frequency bins

for f = 1:file_count
    file_name = data_files(f).name;
    file_name = fullfile('Trainings', file_name);
    fprintf('Processing file: %s\n', file_name);
    
    load(file_name);
    data_copy = data; % Keep a copy of original data
    nrun = length(data);

    trials_logband = cell(nrun, ntrial);
    labels = cell(nrun, ntrial);
    psd_features = cell(nrun, ntrial);
    baseline_features = cell(nrun, 1); % One baseline per run

    % Bandpass filter (4-56 Hz)
    [b,a] = butter(5, 2*[4 56]/fs, 'bandpass');
    for r = 1:nrun
        data{1,r}.X = filtfilt(b, a, data{1,r}.X);
    end
    
    for r = 1:nrun
        % === Extract Baseline Data (First 5s of Recording - Before Trials) ===
        baseline_data = data{1,r}.X(1:fs*5, :); 
        [baseline_psd, freq_bins] = proc_spectrogram(baseline_data, wlength, wshift, pshift, fs, mlength);
        baseline_features{r} = mean(baseline_psd, 1); % Mean PSD over time

        for p = 1:ntrial
            start_idx = data{1,r}.trial(1,p);
            trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1,:);
            
            % Compute Log-Band Power
            trials_logband{r,p} = log(abs(trial_data).^2);
            
            % Compute Power Spectral Density using proc_spectrogram
            [psd_features{r,p}, ~] = proc_spectrogram(trial_data, wlength, wshift, pshift, fs, mlength);
            
            % Store Labels (1 = right hand, 2 = feet)
            labels{r,p} = data{1,r}.y(p);
        end
    end

    % Store all processed data
    all_trials = [all_trials; trials_logband];
    all_labels = [all_labels; labels];
    all_psd_features = [all_psd_features; psd_features];
    all_baseline_features = [all_baseline_features; baseline_features];
    all_frequencies = freq_bins;
end

%% Fisher Score Calculation (Relative to Baseline)
fprintf('Computing Fisher Score for feature selection...\n');

% Convert PSD features into matrix format
nfeatures = nchannel * length(all_frequencies); 
num_trials = length(all_psd_features);

Features = zeros(num_trials, nfeatures);
BaselineFeatures = zeros(num_trials, nfeatures); 
Labels = zeros(num_trials, 1);

for p = 1:num_trials
    trial_psd = squeeze(mean(all_psd_features{p}, 1)); % Mean across time
    baseline_psd = squeeze(mean(all_baseline_features{p}, 1)); % Mean across time
    
    % Convert PSD to feature vector
    Features(p, :) = reshape(trial_psd', 1, []);
    BaselineFeatures(p, :) = reshape(baseline_psd', 1, []);

    % Store corresponding labels
    Labels(p) = all_labels{p};
end

% Compute Fisher Score relative to Baseline
fprintf('Calculating Fisher Scores with Baseline Comparison...\n');
mean_baseline = mean(BaselineFeatures, 1);
std_baseline = std(BaselineFeatures, 1);

mean_task = mean(Features, 1);
std_task = std(Features, 1);

FisherScores = abs(mean_task - mean_baseline) ./ sqrt(std_task.^2 + std_baseline.^2);
F_map = reshape(FisherScores, [nchannel, length(all_frequencies)]); % Reshape for visualization

% Restrict Fisher Score Map to 4-56 Hz Range
valid_freq_idx = (all_frequencies >= 4 & all_frequencies <= 56);  % Get valid frequency indices
filtered_frequencies = all_frequencies(valid_freq_idx);  % Keep only valid frequencies
filtered_F_map = F_map(:, valid_freq_idx);  % Apply the filter to the Fisher Score matrix

%% Verify Correct Channel-Frequency Mapping
figure;
imagesc(filtered_frequencies, 1:nchannel, filtered_F_map);
colorbar;
xlabel('Frequency (Hz)');
ylabel('Channels');
title('Verified Fisher Score Map (Click to Select Features)');
yticks(1:nchannel);
yticklabels(channel_labels); % Ensure correct channel mapping

% Allow user to select features interactively
fprintf('Click on 5 features in the heatmap to select them for classification.\n');
[x, y] = ginput(3); % Get user clicks
% best combination : Accuracy on train data 62%
% Channel: C4, Frequency: 16.00 Hz, Fisher Score: 0.448
% Channel: CPz, Frequency: 16.00 Hz, Fisher Score: 0.468
% Channel: CP3, Frequency: 12.00 Hz, Fisher Score: 0.408

% Process selected points
selected_features_manual = cell(length(x), 2);
selected_idx = zeros(length(x), 1);
channel_idx = zeros(length(x), 1);
freq_idx = zeros(length(x), 1);

for i = 1:length(x)
    [~, freq_idx(i)] = min(abs(filtered_frequencies - x(i))); % Find closest valid frequency index
    channel_idx(i) = round(y(i)); % Get corresponding channel index
    selected_features_manual{i, 1} = channel_labels{channel_idx(i)};
    selected_features_manual{i, 2} = filtered_frequencies(freq_idx(i));

    % Convert to Fisher Score index in original data
    original_freq_idx = find(all_frequencies == filtered_frequencies(freq_idx(i)), 1);
    selected_idx(i) = sub2ind([nchannel, length(all_frequencies)], channel_idx(i), original_freq_idx);
end

% Display Selected Features
fprintf('Selected Features:\n');
for i = 1:length(selected_idx)
    fprintf('Channel: %s, Frequency: %.2f Hz, Fisher Score: %.3f\n', ...
        selected_features_manual{i, 1}, selected_features_manual{i, 2}, FisherScores(selected_idx(i)));
end

%% Train Gaussian Classifier on Selected Features
fprintf('Training Gaussian Classifier...\n');

% Extract corresponding features from dataset
selected_features = Features(:, selected_idx);

% Split Data (80% Training, 20% Testing)
num_train = round(0.8 * num_trials);
train_data = selected_features(1:num_train, :);
train_labels = Labels(1:num_train);

test_data = selected_features(num_train+1:end, :);
test_labels = Labels(num_train+1:end);

% Train Gaussian Model
gaussian_model = fitcnb(train_data, train_labels);

% Evaluate Classifier
predicted_labels = predict(gaussian_model, test_data);
accuracy = sum(predicted_labels == test_labels) / length(test_labels) * 100;

fprintf('Gaussian Classifier Accuracy: %.2f%%\n', accuracy);
save('Trained_Classifier', 'gaussian_model')

%% Load and Preprocess Evaluation Data
eval_files = dir(fullfile('Evaluations', '*.mat')); % Get all evaluation files
num_eval_files = length(eval_files);

if num_eval_files == 0
    error('No evaluation files found in Evaluations/*.mat');
end

fprintf('Found %d evaluation files. Loading and processing...\n', num_eval_files);

all_eval_features = {}; % Store extracted features
all_eval_labels = {}; % Store true labels

for f = 1:num_eval_files
    eval_file = eval_files(f).name;
    eval_path = fullfile('Evaluations', eval_file);
    fprintf('Processing evaluation file: %s\n', eval_path);
    
    load(eval_path);
    nrun = length(data);

    eval_psd_features = cell(nrun, ntrial);
    eval_labels = cell(nrun, ntrial);

    % Apply same bandpass filter as training data
    [b,a] = butter(5, 2*[4 56]/fs, 'bandpass');
    for r = 1:nrun
        data{1,r}.X = filtfilt(b, a, data{1,r}.X);
    end
    
    for r = 1:nrun
        for p = 1:ntrial
            start_idx = data{1,r}.trial(1,p);
            trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1,:);

            % Compute Power Spectral Density using proc_spectrogram
            [eval_psd_features{r,p}, ~] = proc_spectrogram(trial_data, wlength, wshift, pshift, fs, mlength);
            
            % Store true labels (1 = right hand, 2 = both feet)
            eval_labels{r,p} = data{1,r}.y(p);
        end
    end

    % Store extracted features and labels
    all_eval_features = [all_eval_features; eval_psd_features];
    all_eval_labels = [all_eval_labels; eval_labels];
end

%% Extract Features Using Selected Indices from Training Step
num_eval_trials = length(all_eval_features);
eval_selected_features = zeros(num_eval_trials, length(selected_idx)); % Ensure correct size

for p = 1:num_eval_trials
    trial_psd = squeeze(mean(all_eval_features{p}, 1)); % Mean across time

    % Correctly extract selected features (one by one)
    for i = 1:length(selected_idx)
        eval_selected_features(p, i) = trial_psd(channel_idx(i), freq_idx(i)); % Extract single value
    end
end

eval_labels_vector = cell2mat(all_eval_labels); % Convert labels to vector

%% Evaluate Classifier Performance
fprintf('Evaluating Gaussian Classifier on New Data...\n');

% Predict labels on evaluation data
predicted_eval_labels = predict(gaussian_model, eval_selected_features);

% Compute accuracy
eval_accuracy = sum(predicted_eval_labels == eval_labels_vector') / num_eval_trials * 100;
fprintf('Evaluation Accuracy: %.2f%%\n', eval_accuracy);

%% Plot Confusion Matrix
eval_labels_vector = eval_labels_vector(:); % Convert to 1D column vector
predicted_eval_labels = predicted_eval_labels(:); % Convert to 1D column vector
% Check if the number of trials is correct
num_eval_trials = length(predicted_eval_labels);

% Extract only the first num_eval_trials labels from eval_labels_vector
eval_labels_vector = eval_labels_vector(1:num_eval_trials); 


% Check if the sizes match before computing the confusion matrix
if length(eval_labels_vector) ~= length(predicted_eval_labels)
    error('Mismatch: The number of true labels (%d) and predicted labels (%d) do not match.', ...
          length(eval_labels_vector), length(predicted_eval_labels));
end

%Compute Confusion Matrix
conf_mat = confusionmat(eval_labels_vector, predicted_eval_labels); % Compute confusion matrix

figure;
confusionchart(conf_mat);
title('Confusion Matrix for Evaluation Data');


