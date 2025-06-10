%% Diagnostic Loader for BIDS-Structured XDF Files
addpath("Matlab-xdf/")
root_dir = 'Recordings';
xdf_files = dir(fullfile(root_dir, '**', '*.xdf'));

fprintf('\n🔍 Found %d .xdf files\n', length(xdf_files));

for i = 1:length(xdf_files)
    file_info = xdf_files(i);
    full_path = fullfile(file_info.folder, file_info.name);
    fprintf('\n🧠 Loading [%02d/%02d]: %s\n', i, length(xdf_files), full_path);

    try
        [streams, ~] = load_xdf(full_path);
        if isempty(streams)
            fprintf('⚠️  No streams found. Skipping.\n');
            continue
        end

        % Find EEG stream
        eeg_stream = [];
        for s = 1:length(streams)
            if contains(streams{s}.info.type, 'EEG', 'IgnoreCase', true)
                eeg_stream = streams{s};
                break
            end
        end

        if isempty(eeg_stream)
            fprintf('⚠️  No EEG stream found. Skipping.\n');
            continue
        end

        eeg_data = eeg_stream.time_series;
        eeg_time = eeg_stream.time_stamps;

        % Diagnostic info
        fprintf('Data shape: %d samples × %d channels\n', size(eeg_data, 1), size(eeg_data, 2));
        fprintf(' Duration: %.2f seconds\n', eeg_time(end) - eeg_time(1));

        % Check for empty stream
        if isempty(eeg_data) || size(eeg_data, 1) < 10
            fprintf('⚠️  EEG data is empty or too short. Skipping.\n');
            continue
        end

        % Quick plot of first 2 channels for 2 seconds
        figure('Name', file_info.name, 'NumberTitle', 'off');
        max_duration = 30; % seconds
        t_idx = find(eeg_time < eeg_time(1) + max_duration);
        t_idx = t_idx(t_idx <= size(eeg_data,1)); % prevent out-of-bounds

        plot(eeg_time(t_idx), eeg_data(t_idx, 1:min(2, size(eeg_data,2))));

        title(sprintf('EEG Preview – %s', file_info.name), 'Interpreter', 'none');
        xlabel('Time (s)'); ylabel('Amplitude (µV)');
        legend({'Ch1', 'Ch2'});
        drawnow;

    catch ME
        fprintf('💥 ERROR: %s\n', ME.message);
        continue
    end
end


