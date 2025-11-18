function cost = ann_pso_cost_function(weights, net, inputs, targets)
    % ANN_PSO_COST_FUNCTION Evaluates the performance of the ANN for a given set of weights.
    %
    % cost = ANN_PSO_COST_FUNCTION(weights, net, inputs, targets)
    %   weights - A row vector of weights and biases for the network.
    %   net     - The neural network structure.
    %   inputs  - Input data for the network (features x samples).
    %   targets - Target data for the network (outputs x samples).
    
    % Set the network's weights and biases
    net = setwb(net, weights');
    
    % Perform the simulation (prediction)
    predictions = net(inputs);
    
    % Calculate the error (cost) - Using Mean Squared Error
    error = predictions - targets;
    cost = mean(error.^2);
    
    % Optional: You could add a regularization term here to fight overfitting
    % e.g., cost = cost + 0.01 * sum(weights.^2); 
end