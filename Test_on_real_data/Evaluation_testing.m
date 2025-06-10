%% Test of the classifier on new data

clc; clear; close all;

%% === Load Trained Classifier ===
load('trained_svm_model.mat', 'svm_model', 'selected_idx', 'nonzero_var_idx');

%% === Load & Preprocess New Data ===
% This should mimic your training pipeline exactly
load('Evaluations/XDF_converted_P001_run5.mat');  % Replace with your new EEG data file
fs = 128;
ntrial = 10;
nchannel = 16;
T = 1/fs;

[b,a] = butter(5, 2*[4 30]/fs, 'bandpass');
for r = 1:length(data)
    data{1,r}.X = filtfilt(b, a, data{1,r}.X);
end

%% === Extract Trials & Log Band Power ===
trials_logband = cell(size(data));
for r = 1:length(data)
    for p = 1:ntrial
        start_idx = data{1,r}.trial(1,p);
        trial_data = data{1,r}.X(start_idx:start_idx+fs*5-1,:);
        trials_logband{r,p} = log(abs(trial_data).^2);
    end
end

%% === CSP Feature Extraction (same as training) ===
X_run = zeros(nchannel, fs*5, ntrial);
y_run = data{1,1}.y;  % or use dummy labels

for p = 1:ntrial
    X_run(:, :, p) = trials_logband{1,p}';
end

% Use same CSP computation
W = csp(X_run, y_run); % CSP filters for the current run

% Extract CSP features
csp_features_new = zeros(ntrial, nchannel);
for p = 1:ntrial
    Z = W' * X_run(:,:,p);
    var_csp = var(Z, 0, 2);
    csp_features_new(p,:) = log(var_csp / sum(var_csp));
end

%% === Apply Feature Selection ===
features_new = csp_features_new(:, selected_idx);
features_new = features_new(:, nonzero_var_idx);

%% === Predict Using Trained Classifier ===
y_pred_new = predict(svm_model, features_new);

% Display predictions
disp('Predicted class labels:');
disp(y_pred_new);

% Setup UDP sender
u = udpport("IPV4");
unityIP = "127.0.0.1";
unityPort = 5013;

% Define message map
for i = 1:length(y_pred_new)
    class_id = y_pred_new(i);
    
    switch class_id
        case 1
            msg = "lift";  % Right hand
        case 2
            msg = "rest";  % right Arm or null command
        otherwise
            warning("Unknown class ID: %d", class_id);
            continue;
    end

    % Send UDP message
    try
        write(u, msg, "string", unityIP, unityPort);
        fprintf('[%s] Sent to Unity: %s\n', datestr(now, 'HH:MM:SS'), msg);
    catch err
        warning("UDP send failed: %s", err.message);
    end
    
    pause(6); % Delay between messages
end