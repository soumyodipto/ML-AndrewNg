function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

J = 0;
for i = 1:m
    J+=(-y(i,1)*log(sigmoid(X(i,:)*theta))-(1-y(i,1))*log(1-sigmoid(X(i,:)*theta)));    
end
J/=m;
%J+=(lambda*0.5/m)*theta(2:,:)

for i = 1:size(theta)(1)
    if i > 1
        gradsum = 0;
        for j = 1:m
            gradsum += ((sigmoid(X(j,:)*theta)-y(j,1))*X(j,i));
        end
        grad(i,1) = (gradsum/m + (lambda/m)*theta(i,1));
        J += ((lambda*0.5/m)*(theta(i,1)**2));       
    else
        gradsum = 0;
        for j = 1:m
            gradsum += ((sigmoid(X(j,:)*theta)-y(j,1))*X(j,i));
        end
	grad(i,1) = gradsum/m;
end



% =============================================================

end
