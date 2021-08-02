import numpy as np
 
class DerivativeRankScaler():
    """ Scales by ranking the derivative of original data

    d: order of derivative
    epsilon: should be zero, values closer than epsilon are considered equal
    """
    def __init__(self, d=1, epsilon=0):
        self.d = d
        self.epsilon = epsilon
    def fit(self,X):
        """
        unsqueeze if input is 1-dim
        clean input and keep as X_fit
        apply the scaling to each column of the input to obtain Y_fit,
        """
        if len(X.shape)==1:
            X=X[:,np.newaxis]
        elif(len(X.shape)>2):
            raise ValueError("input shape not supported")
        self.X_fit = np.apply_along_axis(self._clean_input,0,X)
        self.Y_fit = np.apply_along_axis(lambda x: self._derivate_rank(x,self.d),0,self.X_fit)
        return self

    def transform(self,X):
        """use the fitted values X_fit and the scaled Y_fit and interpolate to get values for X
        -inf(inf) will be scaled to min(max) of Y_fit, nan stay nan
        """
        if len(X.shape)==1:
            X=np.reshape(X,(X.shape[0],1))
        if (X.shape[1]!=self.X_fit.shape[1]):
            raise ValueError("Expected {} features. Got {}".format(self.X_fit.shape[1],X.shape[1]))
        Y = np.column_stack([np.interp(X[:,i], self.X_fit[:,i], self.Y_fit[:,i]) for i in np.arange(self.X_fit.shape[1])])
        return Y.flatten()

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

    def _clean_input(self,x):
        """
        treat input:
        replace non-finites with arbitrary present value, here max,
        this double value has no effect in fit and in transform interpolate is used
        which maps inf to max, -inf to min, and nan to nan
        the cleaning is done to erase non-finites and to keep shape at the same time
        finally the input is sorted, because X_fit must be sorted for use interp in transform
        """
        x_use = x.copy()
        x_use[np.isfinite(x_use)==False]=np.max(x_use[np.isfinite(x_use)])
        return np.sort(x_use)

    def _reset(self):
        if hasattr(self, 'X_fit'):
            del self.X_fit
            del self.Y_fit

    def _derivate_rank(self,x,d):
        """
        recursively apply the derivative (get step sizes of ordered series)
        if d=0 is reached, set all stepsizes to one (except if step is zero)
        and re-integrate back
        scale the output to absolute max of 1
        """
        delta,order,x0 = self._get_steps(x)
        if d>0:
            delta = self._derivate_rank(delta,d-1)
        re_integrated = self._integrate(delta,order,x0,constant=(d==0))
        return re_integrated/np.max(np.abs(re_integrated))

    def _get_steps(self,x): # get stepsizes after ordering the input vector
        """
        used in _derivate_rank  to get the discrete derivative, i.e., the stepsizes
        output is the order of sorted series, the stepsizes (delta) and smallest value x0 
        """
        order = np.argsort(x)
        delta = np.diff(x[order])
        x0 = x[order][0]
        return delta,order,x0 # return the step sizes, the ordering of input vector and smallest element of x

    def _integrate(self,delta,order,x0,constant=False): # y is non-negative vector of stepsizes to recreate ordered vector x0,x1,...
        """
        to re-integrate, need the deltas, the order related to deltas and the smallest value of integrated series
        then, just add the deltas one-by-one to x0 and finally inverse the order to get the original series
        """
        assert(all(delta>=0))
        if constant: 
            delta = (delta>self.epsilon).astype(int) # make all steps 1 or zero if not above threshold
        x_sorted = x0+np.append(0,np.cumsum(delta))
        return x_sorted[np.argsort(order)] # reverse the order according to order input before returning
