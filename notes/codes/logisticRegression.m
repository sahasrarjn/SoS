function g = sigmoid(z)
	%SIGMOID Compute sigmoid function
	%   g = SIGMOID(z) computes the sigmoid of z.

	% You need to return the following variables correctly 
	g = ones(size(z));
	g = g ./ (1+exp(-z));
end

function [J, grad] = costFunction(theta, X, y)
	%COSTFUNCTION Compute cost and gradient for logistic regression
	%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
	%   parameter for logistic regression and the gradient of the cost
	%   w.r.t. to the parameters.

	% Initialize some useful values
	m = length(y); % number of training examples
	h = sigmoid(X*theta);

	% You need to return the following variables correctly 
	J = 1/m * (-y'*log(h)-(1-y)'*log(1-h));

	grad = 1/m * (h-y)'*X;

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
	%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
	%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
	%   theta as the parameter for regularized logistic regression and the
	%   gradient of the cost w.r.t. to the parameters. 

	% Initialize some useful values
	m = length(y); % number of training examples

	[c,g] = costFunction(theta, X, y);

	temp = lambda/(2*m) * theta'*theta;
	temp = temp - lambda/(2*m)*theta(1,1)^2;
	J = c+temp;


	grad = g + lambda/m * theta';
	grad(1,1) = g(1,1);

end

function out = mapFeature(X1, X2)
	% MAPFEATURE Feature mapping function to polynomial features
	%
	%   MAPFEATURE(X1, X2) maps the two input features
	%   to quadratic features used in the regularization exercise.
	%
	%   Returns a new feature array with more features, comprising of 
	%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

	degree = 6;
	out = ones(size(X1(:,1)));
	for i = 1:degree
	    for j = 0:i
	        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
	    end
	end
end

function plotDecisionBoundary(theta, X, y)
	%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
	%the decision boundary defined by theta
	%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
	%   positive examples and o for the negative examples. X is assumed to be 
	%   a either 
	%   1) Mx3 matrix, where the first column is an all-ones column for the 
	%      intercept.
	%   2) MxN, N>3 matrix, where the first column is all-ones

	% Plot Data
	plotData(X(:,2:3), y);
	hold on

	if size(X, 2) <= 3
	    % Only need 2 points to define a line, so choose two endpoints
	    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

	    % Calculate the decision boundary line
	    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

	    % Plot, and adjust axes for better viewing
	    plot(plot_x, plot_y)
	    
	    % Legend, specific for the exercise
	    legend('Admitted', 'Not admitted', 'Decision Boundary')
	    axis([30, 100, 30, 100])
	else
	    % Here is the grid range
	    u = linspace(-1, 1.5, 50);
	    v = linspace(-1, 1.5, 50);

	    z = zeros(length(u), length(v));
	    % Evaluate z = theta*x over the grid
	    for i = 1:length(u)
	        for j = 1:length(v)
	            z(i,j) = mapFeature(u(i), v(j))*theta;
	        end
	    end
	    z = z'; % important to transpose z before calling contour

	    % Plot z = 0
	    % Notice you need to specify the range [0, 0]
	    contour(u, v, z, [0, 0], 'LineWidth', 2)
	end
	hold off

end

function p = predict(theta, X)
	%PREDICT Predict whether the label is 0 or 1 using learned logistic 
	%regression parameters theta
	%   p = PREDICT(theta, X) computes the predictions for X using a 
	%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

	m = size(X, 1); % Number of training examples

	% You need to return the following variables correctly
	p = zeros(m, 1);

	p = sigmoid(X*theta);

	for i = [1:m],
	  if p(i,1) >= 0.5,
	    p(i,1) = 1;
	   else
	    p(i,1) = 0;
	  end
	end
end



