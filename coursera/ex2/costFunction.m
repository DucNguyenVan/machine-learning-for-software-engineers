function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

   sum_1 = 0;
    sum_2 = 0;
    sum_3 = 0;
for j=1:m
  sum_1 = sum_1 - (sigmoid(theta(1)*X(j, 1) + theta(2)*X(j,2) + theta(3)*X(j,3)) - y(j))*X(j, 1)
  sum_2 = sum_2 - (sigmoid(theta(1)*X(j, 1) + theta(2)*X(j,2) + theta(3)*X(j,3)) - y(j))*X(j, 2)
  sum_3 = sum_3 - (sigmoid(theta(1)*X(j, 1) + theta(2)*X(j,2) + theta(3)*X(j,3)) - y(j))*X(j, 3)
end

grad(1) = grad(1) - 1/m*sum_1
grad(2) = grad(2) - 1/m*sum_2
grad(3) = grad(3) - 1/m*sum_3


for j=1:m
  hx = sigmoid(theta(1)*X(j, 1) + theta(2)*X(j,2) + theta(3)*X(j,3))
  J = J + ( y(j)*log(hx) + (1 - y(j))*log(1 - hx) )
end
J = -J/m




% =============================================================

end
