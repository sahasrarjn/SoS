function [mu sigma2] = estimateGaussian(X)
	%ESTIMATEGAUSSIAN This function estimates the parameters of a 
	%Gaussian distribution using the data in X
	%   [mu sigma2] = estimateGaussian(X), 
	%   The input X is the dataset with each n-dimensional data point in one row
	%   The output is an n-dimensional vector mu, the mean of the data set
	%   and the variances sigma^2, an n x 1 vector
	% 

	% Useful variables
	[m, n] = size(X);

	% You should return these values correctly
	mu = zeros(n, 1);
	sigma2 = zeros(n, 1);

	mu = sum(X)'/m;

	mu_repeated = repmat(mu', m, 1);
	temp = X-mu_repeated;

	sigma2 = sum(temp.^2)'/m;
end

% ==============================================

function [bestEpsilon bestF1] = selectThreshold(yval, pval)
	%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
	%outliers
	%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
	%   threshold to use for selecting outliers based on the results from a
	%   validation set (pval) and the ground truth (yval).
	%

	bestEpsilon = 0;
	bestF1 = 0;
	F1 = 0;

	stepsize = (max(pval) - min(pval)) / 1000;
	for epsilon = min(pval):stepsize:max(pval)

	    predictions = (pval < epsilon);

	    fp = sum((predictions==1) & (yval==0));
	    fn = sum((predictions==0) & (yval==1));
	    tp = sum((predictions==1) & (yval==1));
	    tn = sum((predictions==0) & (yval==0));

	    prec = tp / (tp + fp);
	    rec  = tp / (tp + fn);

	    F1 = 2*prec*rec/(prec+rec);


	    if F1 > bestF1
	       bestF1 = F1;
	       bestEpsilon = epsilon;
	    end
	end

end
% =================================================

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
	%COFICOSTFUNC Collaborative filtering cost function
	%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
	%   num_features, lambda) returns the cost and gradient for the
	%   collaborative filtering problem.
	%

	% Unfold the U and W matrices from params
	X = reshape(params(1:num_movies*num_features), num_movies, num_features);
	Theta = reshape(params(num_movies*num_features+1:end), ...
	                num_users, num_features);

	            
	% You need to return the following values correctly
	J = 0;
	X_grad = zeros(size(X));
	Theta_grad = zeros(size(Theta));

	temp = X*Theta' - Y;


	J = sum(sum(R.*(temp.^2))) + lambda*(sum(sum(X)) + sum(sum(Theta))); 
	J /= 2;

	X_grad = ((R.*temp)*Theta);
	Theta_grad = ((R.*temp)'*X);

	R
	X

	% ===============================================

	grad = [X_grad(:); Theta_grad(:)];

end
