function [J, grad] = lrCostFunction(theta, X, y, lambda)
	%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
	%regularization
	%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
	%   theta as the parameter for regularized logistic regression and the
	%   gradient of the cost w.r.t. to the parameters. 

	% Initialize some useful values
	m = length(y); % number of training examples

	% You need to return the following variables correctly 
	J = 0;
	grad = zeros(size(theta));

	h = sigmoid(X*theta);

	temp = theta;
	temp(1) = 0;

	J = 1/m * (-log(h)'*y-log(1-h)'*(1-y)) + lambda/(2*m) * temp'*temp;

	grad = 1/m * X'*(h-y) + lambda/m * temp;

	grad = grad(:);
end
% =============================================================

function [all_theta] = oneVsAll(X, y, num_labels, lambda)
	%ONEVSALL trains multiple logistic regression classifiers and returns all
	%the classifiers in a matrix all_theta, where the i-th row of all_theta 
	%corresponds to the classifier for label i
	%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
	%   logistic regression classifiers and returns each of these classifiers
	%   in a matrix all_theta, where the i-th row of all_theta corresponds 
	%   to the classifier for label i

	% Some useful variables
	m = size(X, 1);
	n = size(X, 2);

	all_theta = zeros(num_labels, n + 1);

	% Add ones to the X data matrix
	X = [ones(m, 1) X];

	for c = (1:num_labels),
	  initial_theta = zeros(n+1,1);
	  options = optimset('GradObj', 'on', 'MaxIter', 50);
	  [theta] = ...
	      fmincg (@(t)(lrCostFunction(t, X, (y==c), lambda)), initial_theta, options);
	  
	  all_theta(c,:) += theta';
	end
end
% =============================================================


function p = predictOneVsAll(all_theta, X)
	%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
	%are in the range 1..K, where K = size(all_theta, 1). 
	%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
	%  for each example in the matrix X. Note that X contains the examples in
	%  rows. all_theta is a matrix where the i-th row is a trained logistic
	%  regression theta vector for the i-th class. You should set p to a vector
	%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
	%  for 4 examples) 

	m = size(X, 1);
	num_labels = size(all_theta, 1);

	p = zeros(size(X, 1), 1);

	% Add ones to the X data matrix
	X = [ones(m, 1) X];

	temp = sigmoid(X*all_theta');
	[~,p] = max(temp, [], 2);

end
% =============================================================


function p = predict(Theta1, Theta2, X)
	%PREDICT Predict the label of an input given a trained neural network
	%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	%   trained weights of a neural network (Theta1, Theta2)

	% Useful values
	m = size(X, 1);
	num_labels = size(Theta2, 1);

	p = zeros(size(X, 1), 1);

	X = [ones(m, 1), X];

	z2 = sigmoid(X*Theta1');
	z2 = [ones(m, 1), z2];

	z3 = sigmoid(z2*Theta2');

	[~, p] = max(z3, [], 2);

end
% =========================================================================