function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 


% W1 is a vector which is of size (number of input units)*(number of hidden units)
%     (the weights from each input to each hidden unit)
% b1 is a vector of size (number of hidden units)
%     (the bias of each hidden unit)
% W2 and b2 are the same, but for the connection from the hidden units to the output units

rho = sparsityParam;

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% make a 1's row vector
numSamples = size(data,2);
rowVecOnes = ones(1,numSamples);

%%%%%%%%%%%%%%%%%%%
% FORWARD PASS section begin
%%%%%%%%%%%%%%%%%%
%
% Get the input and output of the hidden layer (output is the "activiation" of that layer)
hiddenInputs = W1*data + b1*rowVecOnes;  %(25x64)*(64x1000) + (25x1)*(1x1000)   = (25x1000) + (25x1000) = 25x1000
hiddenOutputs = sigmoid(hiddenInputs);   %(25x1000)

%
% Get the input and output of the output layer (output is the "activiation" of that layer)
finalInputs = W2*hiddenOutputs + b2*rowVecOnes; %  (64x25)*(25x1000) + (64x1)(1x1000) = 64x1000
outputs = sigmoid(finalInputs);   % 64x1000

%
% Calculate the average activation of the hidden units for the sparsity constraint
rho_hiddenOutputs = (1.0/numSamples) * sum(hiddenOutputs,2);  % should be 25x1

%%%%%%%%%%%%%%%%%%%
% FORWARD PASS section end
%%%%%%%%%%%%%%%%%%

%
% NOTE: Layer 1 = input layer
%       Layer 2 = hidden layer
%       Layer 3 = output layer



%
% Now calculate the error terms for the output layer
fprimeOutput = outputs.*(1-outputs);    % element wise ==> result is 64x1000 matrix
deltaOutput = -(data - outputs) .* fprimeOutput;    % element wise ==> result is 64x1000 matrix
%
% Now calculate the error terms for the hidden layer
fprimeHidden = hiddenOutputs.*(1-hiddenOutputs);   % element wise ==> result is 25x1000 matrix
deltaHidden = (transpose(W2)*deltaOutput + beta * (-rho./rho_hiddenOutputs + ((1-rho)./(1.0-rho_hiddenOutputs)))*rowVecOnes).*fprimeHidden;  % (25x64)*(64x1000) .* (25x1000) = (25x1000)

% 
% Now calculate the partial derivatives
delW1 = deltaHidden * transpose(data);   % (25x1000)*(1000x64) = 25x64
%delW2 = deltaOutput * transpose(hiddenOutputs); % (64x1000)*(1000x25) = 64x25

delW2 = deltaOutput * transpose(hiddenOutputs)  ; % (64x1000)*(1000x25) = 64x25

delb1 = deltaHidden * transpose(rowVecOnes);     % (25x1000)*(1000x1) = 25x1
delb2 = deltaOutput * transpose(rowVecOnes);     % (64x1000)*(1000x1) = 64x1

% Now sum
W1grad = delW1/numSamples + lambda*W1;
W2grad = delW2/numSamples + lambda*W2;
b1grad = delb1/numSamples;
b2grad = delb2/numSamples;
%
% Calculate cost function
error = (outputs - data); % (64x1000)

errorSQ = error.*error;   % (64x1000)

cost = (0.5/numSamples)*sum(sum(errorSQ));  % sum both rows and columns




%
% Now add in the regularization term
regularization = 0.5*lambda*( sum(sum(W1.*W1)) + sum(sum(W2.*W2)) ); 

cost = cost + regularization;

%
% Add the sparsity constraint to the cost
sparseTerm = rho * log(rho ./ rho_hiddenOutputs) + (1-rho) * log((1.0-rho)./(1.0-rho_hiddenOutputs));
cost = cost + beta*sum(sparseTerm);

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

