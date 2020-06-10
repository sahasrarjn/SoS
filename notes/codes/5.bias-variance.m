function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
	%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
	%regression with multiple variables
	%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
	%   cost of using theta as the parameter for linear regression to fit the 
	%   data points in X and y. Returns the cost in J and the gradient in grad

	% Initialize some useful values
	m = length(y); % number of training examples

	% You need to return the following variables correctly 
	J = 0;
	grad = zeros(size(theta));

	h = X*theta;

	J = (h-y)'*(h-y);

	theta(1)=0;

	J += lambda * theta'*theta;
	J /= 2*m;


	grad = X'*(h-y);
	grad += lambda*theta;
	grad /= m;



	grad = grad(:);

end

% ==============================================

function [error_train, error_val] = ...
	    learningCurve(X, y, Xval, yval, lambda)
	%LEARNINGCURVE Generates the train and cross validation set errors needed 
	%to plot a learning curve
	%   [error_train, error_val] = ...
	%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
	%       cross validation set errors for a learning curve. In particular, 
	%       it returns two vectors of the same length - error_train and 
	%       error_val. Then, error_train(i) contains the training error for
	%       i examples (and similarly for error_val(i)).
	%
	%   In this function, you will compute the train and test errors for
	%   dataset sizes from 1 up to m. In practice, when working with larger
	%   datasets, you might want to do this in larger intervals.
	%

	% Number of training examples
	m = size(X, 1);

	% You need to return these values correctly
	error_train = zeros(m, 1);
	error_val   = zeros(m, 1);

	
	for i = 1:m
	   Xtemp = X(1:i, :);
	   ytemp = y(1:i);

	   theta = trainLinearReg(Xtemp, ytemp, lambda);


	  error_train(i) = linearRegCostFunction(Xtemp, ytemp, theta, 0);
	  error_val(i) = linearRegCostFunction(Xval, yval, theta, 0); 
	end

end

% ============================================

function [X_poly] = polyFeatures(X, p)
	%POLYFEATURES Maps X (1D vector) into the p-th power
	%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
	%   maps each example into its polynomial features where
	%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
	%


	% You need to return the following variables correctly.
	X_poly = zeros(numel(X), p);

	X_poly(:,1) = X;

	for i = 2:p,
	  X_poly(:,i) = X_poly(:,i-1) .* X; 
	end

end
% ===============================================

function [lambda_vec, error_train, error_val] = ...
	    validationCurve(X, y, Xval, yval)
	%VALIDATIONCURVE Generate the train and validation errors needed to
	%plot a validation curve that we can use to select lambda
	%   [lambda_vec, error_train, error_val] = ...
	%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
	%       and validation errors (in error_train, error_val)
	%       for different values of lambda. You are given the training set (X,
	%       y) and validation set (Xval, yval).
	%

	% Selected values of lambda (you should not change this)
	lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

	% You need to return these variables correctly.
	error_train = zeros(length(lambda_vec), 1);
	error_val = zeros(length(lambda_vec), 1);

	for i = 1:length(lambda_vec)
	   lambda = lambda_vec(i);

	   % Compute train / val errors when training linear 
	   % regression with regularization parameter lambda
	   % You should store the result in error_train(i)
	   % and error_val(i)
	   ....
	   theta = trainLinearReg(X, y, lambda);
	   error_train(i) = linearRegCostFunction(X, y, theta, 0);
	   error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
	   
	end
end
% =================================================