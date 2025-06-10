%% Visual Cue Presentation for Motor Imagery (Right Hand vs Feet)
clear; clc; close all;

%% Parameters
n_trials = 10;              % Total number of trials, approximately 2 minutes
cue_time = 3;               % Duration of fixation cross (s)
duration = 5;               % Motor imagery task cue duration (s)
rest_time = 2;              % Rest between trials (s)
feedback_duration = 1;      % Feedback phase duration (s)

% Class labels
classes = {'Right Hand', 'Right Arm'};

%% GUI Setup
fig = figure('Color', 'black', 'Position', [100, 100, 600, 400]);
cue_text = uicontrol('Style', 'text', 'String', '', 'FontSize', 40, ...
    'ForegroundColor', 'white', 'BackgroundColor', 'black', ...
    'Position', [150, 150, 300, 100]);

disp('Starting Visual Cue Presentation...');
pause(1); % Optional pause before starting

%% Trial Loop
for trial = 1:n_trials
    % Fixed alternation: odd trials = Right Hand, even = Feet
    task = mod(trial+1, 2) + 1; % 1 → Right Hand, 2 → Feet

    % Show fixation cross
    cue_text.String = '+';
    pause(cue_time);

    % Show motor imagery cue
    cue_text.String = classes{task};
    fprintf('Trial %d: %s\n', trial, classes{task});
    pause(duration);

    % Show feedback
    cue_text.String = 'Feedback...';
    pause(feedback_duration);

    % Rest period
    cue_text.String = '';
    pause(rest_time);
end

disp('✅ Session Completed.');
close(fig);
