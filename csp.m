function [W] = csp(data, labels)
    % CSP - Common Spatial Pattern algorithm for EEG data
    % INPUT:
    %   data   - EEG data matrix (channels x samples x trials)
    %   labels - Corresponding class labels (1 or 2) [n_trials x 1]
    % OUTPUT:
    %   W      - Spatial filter matrix (channels x channels)
    
    [channels, samples, trials] = size(data); % Get matrix dimensions

    % Separate trials by class
    class1_idx = find(labels == 1);
    class2_idx = find(labels == 2);

    % Error handling if one class is missing
    if isempty(class1_idx) || isempty(class2_idx)
        error('CSP requires at least one trial from each class.');
    end

    % Compute covariance matrices for each class
    cov1 = zeros(channels, channels);
    cov2 = zeros(channels, channels);
    
    for i = 1:length(class1_idx)
        idx = class1_idx(i);
        trial_data = data(:,:,idx); % Extract trial (channels x samples)
        C = (trial_data * trial_data') / trace(trial_data * trial_data'); % Normalize
        cov1 = cov1 + C;
    end
    cov1 = cov1 / length(class1_idx); % Average over all trials

    for i = 1:length(class2_idx)
        idx = class2_idx(i);
        trial_data = squeeze(data(:,:,idx)); % Extract trial (channels x samples)
        C = (trial_data * trial_data') / trace(trial_data * trial_data'); % Normalize
        cov2 = cov2 + C;
    end
    cov2 = cov2 / length(class2_idx);

    % Regularization to prevent singular matrices
    eps_val = eps * eye(channels);
    cov1 = cov1 + eps_val;
    cov2 = cov2 + eps_val;

    % Solve generalized eigenvalue problem
    [V, D] = eig(cov1, cov1 + cov2);

    % Sort eigenvalues in descending order
    [~, order] = sort(diag(D), 'descend');
    W = V(:, order); % Sorted eigenvectors form the CSP projection matrix
end
