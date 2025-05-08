%% Convert Single XDF Run to Training .mat Format (Protocol-aware)
clear all; clc; close all

addpath("Matlab-xdf/");
fs = 128;
mi_duration = 5;            % seconds
cue_time = 3;               % seconds before MI
feedback_time = 1;          % seconds
rest_time = 2;              % seconds
total_trial_time = cue_time + mi_duration + feedback_time + rest_time;

% Load the .xdf file
file_path = 'Recordings/sub-P001/ses-S010/eeg/sub-P001_ses-S010_task-Default_run-001_eeg.xdf';
[streams, ~] = load_xdf(file_path);

% Find EEG stream
eeg = [];
for i = 1:length(streams)
    if contains(streams{i}.info.type, 'EEG', 'IgnoreCase', true)
        eeg = streams{i};
        break;
    end
end

if isempty(eeg)
    error('No EEG stream found.');
end

eeg_data = eeg.time_series;         % [samples × channels]
eeg_time = eeg.time_stamps;
eeg_time = eeg_time(:,fs*4:end);

% Transpose to [samples x channels] if needed
if size(eeg_data, 1) < size(eeg_data, 2)
    eeg_data = eeg_data';
end

eeg_data = eeg_data(4*fs:end, :);  
nchannel = size(eeg_data, 2);

% Determine how many trials we can fit in the recording
total_samples = size(eeg_data, 1);
n_trials = floor(total_samples / (fs * total_trial_time));

if n_trials < 1
    error('Not enough data to extract even one full trial.');
end

% Calculate start of each MI trial (after the cue time)
trial = zeros(1, n_trials);
for i = 1:n_trials
    t_sec = (i - 1) * total_trial_time + cue_time;
    trial(i) = round(t_sec * fs) + 1;
end

% Alternate task labels: 1 = Right Hand, 2 = Feet
y = repmat([1 2], 1, ceil(n_trials / 2));
y = y(1:n_trials);  % Trim if needed

% Package into .mat format
data = cell(1, 1);  % One run
data{1,1}.X = eeg_data;
data{1,1}.trial = trial; %the value is the start of the trial
data{1,1}.y = y;

% Save
if ~exist('Trainings', 'dir'), mkdir('Trainings'); end
save('Trainings/XDF_converted_P001_run5.mat', 'data');
fprintf('Saved protocol-aligned run to Trainings/XDF_converted_run1.mat\n');
