function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vec = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec  = [ 0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
predictions = zeros(size(Xval,1), length(sigma_vec));
accuracy = zeros (1, length(sigma_vec));
accuracy_max = zeros (length(C_vec));
sigma_idx = zeros (length(sigma_vec));
opt_sigma = zeros (length(C_vec), 2);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%       
%

for i=1:length(C_vec),
  for j=1:length(sigma_vec),
model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j))); 
predictions (:, j)  = svmPredict(model, Xval);
accuracy (1,j) = mean(double(predictions (:,j) == yval));
  end
[accuracy_max(i), sigma_idx(i)]= max(accuracy);
opt_sigma (i,:) = [accuracy_max(i) sigma_vec(sigma_idx(i))];
end

[val , k] = max(opt_sigma(:,1));
sigma = opt_sigma(k, 2);
C = C_vec(k);

% =========================================================================

end
