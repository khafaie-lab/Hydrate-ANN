function [net, problem] = create_ann_for_pso(inputs, targets, hiddenLayerSize)
    % CREATE_ANN_FOR_PSO Creates a neural network and defines the optimization problem.
    %
    % [net, problem] = CREATE_ANN_FOR_PSO(inputs, targets, hiddenLayerSize)
    %   inputs - Input data (samples x features)
    %   targets - Target data (samples x outputs)
    %   hiddenLayerSize - A row vector defining the size of hidden layers
    
    % Create a standard feedforward network
    net = feedforwardnet(hiddenLayerSize);
    
    % Configure the network for the data
    % This step is important to get the correct number of weights and biases.
    net = configure(net, inputs', targets');
    
    % Extract the number of parameters (weights and biases)
    problem.nVar = length(getwb(net));
    
    % Define the search space bounds for PSO
    problem.VarMin = -1 * ones(1, problem.nVar);
    problem.VarMax =  1 * ones(1, problem.nVar);
end