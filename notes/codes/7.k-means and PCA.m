function [U, S] = pca(X)
	%PCA Run principal component analysis on the dataset X
	%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
	%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
	%

	% Useful values
	[m, n] = size(X);

	% You need to return the following variables correctly.
	U = zeros(n);
	S = zeros(n);


	[U, S, V] = svd(1/m * X'*X);


end
% ==================================================


function Z = projectData(X, U, K)
	%PROJECTDATA Computes the reduced data representation when projecting only 
	%on to the top k eigenvectors
	%   Z = projectData(X, U, K) computes the projection of 
	%   the normalized inputs X into the reduced dimensional space spanned by
	%   the first K columns of U. It returns the projected examples in Z.
	%

	% You need to return the following variables correctly.
	Z = zeros(size(X, 1), K);

	Z = X*U(:, 1:K);


end
% =================================================

function X_rec = recoverData(Z, U, K)
	%RECOVERDATA Recovers an approximation of the original data when using the 
	%projected data
	%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
	%   original data that has been reduced to K dimensions. It returns the
	%   approximate reconstruction in X_rec.
	%

	% You need to return the following variables correctly.
	X_rec = zeros(size(Z, 1), size(U, 1));


	X_rec = Z*U(:, 1:K)';


end
% ==================================================

function idx = findClosestCentroids(X, centroids)
	%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
	%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
	%   in idx for a dataset X where each row is a single example. idx = m x 1 
	%   vector of centroid assignments (i.e. each entry in range [1..K])
	%

	% Set K
	K = size(centroids, 1);

	% You need to return the following variables correctly.
	idx = zeros(size(X,1), 1);

	for i=1:size(X,1),
	   dist = zeros(1, K);
	   for c = 1:K,
	     dist(1, c) = sqrt(sum(power((X(i,:)-centroids(c,:)),2)));
	   endfor
	   [~, min_idx] = min(dist);
	   idx(i, 1) = min_idx;
	end

end
% =================================================

function centroids = computeCentroids(X, idx, K)
	%COMPUTECENTROIDS returns the new centroids by computing the means of the 
	%data points assigned to each centroid.
	%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
	%   computing the means of the data points assigned to each centroid. It is
	%   given a dataset X where each row is a single data point, a vector
	%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
	%   example, and K, the number of centroids. You should return a matrix
	%   centroids, where each row of centroids is the mean of the data points
	%   assigned to it.
	%

	% Useful variables
	[m n] = size(X);

	% You need to return the following variables correctly.
	centroids = zeros(K, n);

	count = zeros(K,1);

	for i=1:m,
	  centroids(idx(i),:) += X(i,:);
	  count(idx(i)) += 1;
	endfor

	for i=1:K,
	  centroids(i,:) /= count(i, 1);
	end

end
% ===================================================


function centroids = kMeansInitCentroids(X, K)
	%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
	%used in K-Means on the dataset X
	%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
	%   used with the K-Means on the dataset X
	%

	% You should return this values correctly
	centroids = zeros(K, size(X, 2));


	randidx = randperm(size(X, 1));
	% Take the first K examples as centroids
	centroids = X(randidx(1:K), :);
	                                                                                                                                                                                                                                                                                                                                                

end
% ===================================================

