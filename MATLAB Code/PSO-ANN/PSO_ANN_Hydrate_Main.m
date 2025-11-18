% =========================================================================
% PSO_ANN_Hydrate_Main.m
% =========================================================================
% Author:      Moosa Khafaie
% Date:        []
% Description: This script trains and evaluates a hybrid Particle Swarm
%              Optimization - Artificial Neural Network (PSO-ANN) model to 
%              predict gas hydrate formation temperature (HFT). The PSO
%              algorithm is used to find the optimal weights and biases of the ANN.
%
% Files Needed: create_ann_for_pso.m, ann_pso_cost_function.m
% =========================================================================

%% 1. Initialization
clc; clear; close all;
fprintf('--- Starting PSO-ANN Hydrate Prediction Model Script ---\n\n');

%% 2. Data Loading and Preprocessing (Same as the ANN script)
try
    data = xlsread('Data1139.xls');
    fprintf('Data loaded successfully.\n');
catch
    error('Could not read "Data1139.xls".');
end

X = data(:, 1:end-1);
Y = data(:, end);
[dataNum, ~] = size(X);

% --- Normalization to [-1, 1] as used in the original functions ---
normalize_fcn = @(v, v_min, v_max) 2 * (v - v_min) ./ (v_max - v_min) - 1;
denormalize_fcn = @(v_n, v_min, v_max) (v_n + 1) .* (v_max - v_min) / 2 + v_min;

[Xn, x_settings] = mapminmax(X'); Xn = Xn';
[Yn, y_settings] = mapminmax(Y'); Yn = Yn';

fprintf('Data has been normalized to the range [-1, 1].\n\n');

%% 3. Data Splitting
rng(42); % For reproducibility
[trainInd, ~, testInd] = dividerand(dataNum, 0.80, 0.0, 0.20);
X_train = Xn(trainInd, :); Y_train = Yn(trainInd, :);
X_test = Xn(testInd, :);   Y_test = Yn(testInd, :);
fprintf('Data split into %d training and %d testing samples.\n\n', size(X_train,1), size(X_test,1));

%% 4. Define Network Architecture and Problem for PSO
hiddenLayerSize = [5]; % A single hidden layer with 5 neurons as in the original code
% The `create_ann_for_pso` function handles getting the number of parameters.
[net, problem] = create_ann_for_pso(X_train, Y_train, hiddenLayerSize);
problem.CostFunction = @(weights) ann_pso_cost_function(weights, net, X_train', Y_train');

%% 5. PSO Algorithm Setup
pso_params.MaxIt = 100;      % Maximum number of iterations
pso_params.nPop = 50;       % Population size (Swarm size) - 250 is very large, 50 is more standard
pso_params.w = 1;           % Inertia weight
pso_params.wdamp = 0.99;    % Inertia weight damping ratio
pso_params.c1 = 1.5;        % Personal learning coefficient
pso_params.c2 = 1.5;        % Global learning coefficient

fprintf('--- Starting PSO Algorithm for ANN Training ---\n');

%% 6. PSO Initialization
particle = repmat(struct, pso_params.nPop, 1);
GlobalBest.Cost = inf;
for i = 1:pso_params.nPop
    % Initialize Position
    particle(i).Position = unifrnd(problem.VarMin, problem.VarMax);
    % Initialize Velocity
    particle(i).Velocity = zeros(size(particle(i).Position));
    % Evaluation
    particle(i).Cost = problem.CostFunction(particle(i).Position);
    % Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    % Update Global Best
    if particle(i).Best.Cost < GlobalBest.Cost
        GlobalBest = particle(i).Best;
    end
end
BestCost = zeros(pso_params.MaxIt, 1); % To store the best cost of each iteration

%% 7. PSO Main Loop
for it = 1:pso_params.MaxIt
    for i = 1:pso_params.nPop
        % Update Velocity
        particle(i).Velocity = pso_params.w * particle(i).Velocity ...
            + pso_params.c1 * rand(size(problem.VarMin)) .* (particle(i).Best.Position - particle(i).Position) ...
            + pso_params.c2 * rand(size(problem.VarMin)) .* (GlobalBest.Position - particle(i).Position);
        
        % Update Position
        particle(i).Position = particle(i).Position + particle(i).Velocity;
        
        % Apply Bounds (prevents particles from flying away)
        particle(i).Position = max(particle(i).Position, problem.VarMin);
        particle(i).Position = min(particle(i).Position, problem.VarMax);
        
        % Evaluation
        particle(i).Cost = problem.CostFunction(particle(i).Position);
        
        % Update Personal Best
        if particle(i).Cost < particle(i).Best.Cost
            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;
            
            % Update Global Best
            if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
            end
        end
    end
    
    % Store the Best Cost of the current iteration
    BestCost(it) = GlobalBest.Cost;
    
    % Damping the inertia weight
    pso_params.w = pso_params.w * pso_params.wdamp;
    
    fprintf('Iteration %d: Best Cost = %f\n', it, BestCost(it));
end
fprintf('--- PSO Training Completed ---\n\n');

%% 8. Construct Final Network and Evaluate
% Set the network's weights to the best solution found by PSO
trainedNet = setwb(net, GlobalBest.Position');

% Evaluate on Training and Testing sets
Y_train_pred_n = trainedNet(X_train');
Y_test_pred_n = trainedNet(X_test');

% De-normalize for interpretation
Y_train_orig = mapminmax('reverse', Y_train', y_settings)';
Y_test_orig = mapminmax('reverse', Y_test', y_settings)';
Y_train_pred = mapminmax('reverse', Y_train_pred_n, y_settings)';
Y_test_pred = mapminmax('reverse', Y_test_pred_n, y_settings)';

% Calculate performance metrics
[~, mse_train] = mse(Y_train_pred_n, Y_train');
[~, mse_test] = mse(Y_test_pred_n, Y_test');
r2_train = 1 - mse_train / mse(Y_train' - mean(Y_train'));
r2_test = 1 - mse_test / mse(Y_test' - mean(Y_test'));
rmse_train = sqrt(mean((Y_train_pred - Y_train_orig).^2)); % RMSE on original scale
rmse_test = sqrt(mean((Y_test_pred - Y_test_orig).^2));

% Display Results Table
fprintf('--- PSO-ANN Performance Evaluation ---\n');
fprintf('Metric       \t Training Set \t Testing Set\n');
fprintf('-------------------------------------------\n');
fprintf('MSE (norm)   \t %.6f \t %.6f\n', mse_train, mse_test);
fprintf('RMSE (K)     \t %.4f K \t\t %.4f K\n', rmse_train, rmse_test);
fprintf('R-Squared (R²)\t %.4f   \t %.4f\n', r2_train, r2_test);
fprintf('-------------------------------------------\n\n');

%% 9. Visualization
% --- Figure 1: PSO Convergence Curve ---
figure('Name', 'PSO Convergence', 'NumberTitle', 'off');
plot(BestCost, 'LineWidth', 2);
xlabel('Iteration', 'FontWeight', 'bold');
ylabel('Best Cost (MSE)', 'FontWeight', 'bold');
title('PSO Convergence Over Iterations', 'FontWeight', 'bold');
grid on;

% --- Figure 2: Predicted vs. Actual Plots ---
figure('Name', 'PSO-ANN Performance', 'NumberTitle', 'off');
% Training Set
subplot(1, 2, 1);
plot(Y_train_orig, Y_train_pred, 'o', 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'black');
hold on; refline(1,0); hold off; grid on; axis tight; daspect([1 1 1]);
title('Training Set', 'FontWeight', 'bold');
xlabel('Actual HFT (K)'); ylabel('Predicted HFT (K)');
% Testing Set
subplot(1, 2, 2);
plot(Y_test_orig, Y_test_pred, 's', 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'black');
hold on; refline(1,0); hold off; grid on; axis tight; daspect([1 1 1]);
title('Testing Set', 'FontWeight', 'bold');
xlabel('Actual HFT (K)'); ylabel('Predicted HFT (K)');

sgtitle('PSO-ANN: Predicted vs. Actual HFT', 'FontWeight', 'bold');

fprintf('--- Script Finished ---\n');