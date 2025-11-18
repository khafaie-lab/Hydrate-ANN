% =========================================================================
% ANN_Hydrate_Prediction_Model.m
% =========================================================================
% Author:      Moosa Khafaie
% Date:        [-]
% Description: This script trains and evaluates a Multi-Layer Perceptron (MLP)
%              Artificial Neural Network (ANN) to predict gas hydrate 
%              formation temperature (HFT). The script follows a standard 
%              machine learning workflow: data loading, preprocessing, 
%              splitting, model training, and performance evaluation.
%
% Affiliation: Department of Chemical Engineering, Mahshahr Branch,
%              Islamic Azad University, Mahshahr, Iran
% =========================================================================

%% 1. Initialization and Environment Setup
% Clear workspace, close all figures, and clear command window for a fresh run
clc;
clear;
close all;

fprintf('--- Starting ANN Hydrate Prediction Model Script ---\n\n');

%% 2. Data Loading
% Load the experimental data from an Excel file.
% The file should be in the same directory as the script or a path should be provided.
% Assumption: The last column is the target (output), and all preceding columns are features (inputs).
try
    data = xlsread('Data1139.xls'); % For compatibility. Use readmatrix in newer MATLAB versions.
    fprintf('Data loaded successfully from "Data1139.xls".\n');
catch
    error('Could not find or read "Data1139.xls". Make sure the file is in the correct path.');
end

% Separate features (X) and target (Y)
X = data(:, 1:end-1);
Y = data(:, end);

% Get data dimensions for later use
[dataNum, inputNum] = size(X);
outputNum = size(Y, 2);

fprintf('Dataset dimensions: %d samples, %d input features, %d output feature.\n\n', dataNum, inputNum, outputNum);

%% 3. Data Preprocessing: Normalization
% Normalize features and target to the range [0, 1] using Min-Max scaling.
% This improves the convergence and stability of the training process.
Xn = zeros(size(X)); % Pre-allocate memory for speed
Yn = zeros(size(Y));

% Store min/max values for de-normalization later
minX = min(X);
maxX = max(X);
minY = min(Y);
maxY = max(Y);

% Define a reusable normalization function handle
normalize_fcn = @(val, min_val, max_val) (val - min_val) ./ (max_val - min_val);

% Apply normalization column by column
for ii = 1:inputNum
   Xn(:, ii) = normalize_fcn(X(:, ii), minX(ii), maxX(ii));
end
for ii = 1:outputNum
   Yn(:, ii) = normalize_fcn(Y(:, ii), minY(ii), maxY(ii));
end 

fprintf('Data has been normalized to the range [0, 1].\n\n');

%% 4. Data Splitting: Training and Testing Sets
% Split the data into a training set and a testing set.
% Using a random permutation ensures that the split is unbiased.
rng(42); % Set the random seed for reproducibility! IMPORTANT for GitHub.

trainRatio = 0.80; % 80% for training
valRatio = 0.0;    % 0% for validation (simple train-test split for this example)
testRatio = 0.20;  % 20% for testing

% Using the modern and more powerful 'dividerand' function
[trainInd, valInd, testInd] = dividerand(dataNum, trainRatio, valRatio, testRatio);

% Create training and testing sets
X_train = Xn(trainInd, :)'; % Transpose for MATLAB neural network toolbox format
Y_train = Yn(trainInd, :)';

X_test = Xn(testInd, :)';
Y_test = Yn(testInd, :)';

fprintf('Data split into %d training samples and %d testing samples.\n\n', length(trainInd), length(testInd));

%% 5. ANN Model Definition
% Define the MLP network architecture. This is a key hyperparameter.
hiddenLayerSize = [9 9 9 9 9]; % Example: Your optimized 5 hidden layers with 9 neurons each
activationFunc_hidden = 'tansig';
activationFunc_output = 'purelin';

% Create the feedforward neural network
% 'trainlm' is Levenberg-Marquardt, the default and often best for this size.
net = fitnet(hiddenLayerSize, 'trainlm'); 

% Set activation functions for each layer explicitly for clarity
for i = 1:numel(hiddenLayerSize)
    net.layers{i}.transferFcn = activationFunc_hidden;
end
net.layers{numel(hiddenLayerSize)+1}.transferFcn = activationFunc_output;

% Configure training parameters (optional, but good practice)
net.trainParam.epochs = 1000;          % Maximum number of epochs to train
net.trainParam.goal = 1e-6;           % Performance goal
net.trainParam.showWindow = true;     % Show the training GUI
net.trainParam.showCommandLine = true; % Show progress in command window

fprintf('ANN architecture defined with %d hidden layers.\n', numel(hiddenLayerSize));

%% 6. Model Training
% Train the network using the training data.
fprintf('--- Starting Model Training ---\n');
[trainedNet, tr_record] = train(net, X_train, Y_train);
fprintf('--- Model Training Completed ---\n\n');

%% 7. Model Evaluation
% Evaluate the performance on both training and testing sets.
% First, get the model's predictions (outputs).
Y_train_pred_n = trainedNet(X_train);
Y_test_pred_n = trainedNet(X_test);

% Define a de-normalization function
denormalize_fcn = @(norm_val, min_val, max_val) norm_val .* (max_val - min_val) + min_val;

% De-normalize the predictions and original data to interpret the results
Y_train_pred = denormalize_fcn(Y_train_pred_n, minY, maxY);
Y_test_pred = denormalize_fcn(Y_test_pred_n, minY, maxY);

Y_train_orig = denormalize_fcn(Y_train, minY, maxY);
Y_test_orig = denormalize_fcn(Y_test, minY, maxY);

% Calculate Performance Metrics
mse_train = perform(trainedNet, Y_train, Y_train_pred_n);
rmse_train = sqrt(mse_train);
r2_train = 1 - (sum((Y_train_orig - Y_train_pred).^2) / sum((Y_train_orig - mean(Y_train_orig)).^2));

mse_test = perform(trainedNet, Y_test, Y_test_pred_n);
rmse_test = sqrt(mse_test);
r2_test = 1 - (sum((Y_test_orig - Y_test_pred).^2) / sum((Y_test_orig - mean(Y_test_orig)).^2));

% Display results in a clean table
fprintf('--- Performance Evaluation ---\n');
fprintf('Metric       \t Training Set \t Testing Set\n');
fprintf('-------------------------------------------\n');
fprintf('MSE          \t %.6f \t %.6f\n', mse_train, mse_test);
fprintf('RMSE (K)     \t %.4f K \t\t %.4f K\n', rmse_train, rmse_test);
fprintf('R-Squared (R²)\t %.4f   \t %.4f\n', r2_train, r2_test);
fprintf('-------------------------------------------\n\n');

%% 8. Visualization
% Create professional plots to visualize the results.

% --- Figure 1: Predicted vs. Actual for Training Set ---
figure('Name', 'Performance on Training Set', 'NumberTitle', 'off');
plot(Y_train_orig, Y_train_pred, 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black');
hold on;
% Plot the perfect agreement line
refline = refline(1, 0);
set(refline, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
hold off;
grid on;
title('Training Set: Predicted vs. Actual', 'FontWeight', 'bold');
xlabel('Actual Experimental HFT (K)', 'FontWeight', 'bold');
ylabel('ANN Predicted HFT (K)', 'FontWeight', 'bold');
legend('Model Predictions', 'Perfect Agreement', 'Location', 'northwest');
axis tight; daspect([1 1 1]); % Set aspect ratio to 1:1

% --- Figure 2: Predicted vs. Actual for Testing Set ---
figure('Name', 'Performance on Testing Set', 'NumberTitle', 'off');
plot(Y_test_orig, Y_test_pred, 's', 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'black');
hold on;
refline = refline(1, 0);
set(refline, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
hold off;
grid on;
title('Testing Set: Predicted vs. Actual', 'FontWeight', 'bold');
xlabel('Actual Experimental HFT (K)', 'FontWeight', 'bold');
ylabel('ANN Predicted HFT (K)', 'FontWeight', 'bold');
legend('Model Predictions', 'Perfect Agreement', 'Location', 'northwest');
axis tight; daspect([1 1 1]);

fprintf('--- Script Finished. Check generated plots. ---\n');

% Note: 'Normalize_Fcn' is not a standard MATLAB function. 
% Assuming it's defined as:
% function y = Normalize_Fcn(x, min_val, max_val)
%     y = (x - min_val) / (max_val - min_val);
% end