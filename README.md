## Custom scaler methods: DerivativeRank and logscaler

two custom scaler methods are defined and shown in scaler_demo.ipynb 

### DerivativeRank
allows to set a suitable tradeoff between fixing the distribution of a feature and preserving information from original data.
It is basically a discrete derivative of rank scaling (e.g. like [pandas rank](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html) or similar to sklearn [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer)). compared to those it preserves more of the information from original data. while rank or quantile transformer produce uniform distribution (or close to uniform in case of repeated values), DerivativeRank produces a distribution which is somewhere between uniform and original. the parameter d, the order of derivative, controls how close the resulting distribution is to the original versus the uniform distribution. 

Implementation as recursive funtion:
1. array of values is ordered and consecutive deltas taken, repeated recursively d times
2. differences are set to constant value 1
3. integrating back d times, each time inversing the permutation of ordering applied in delta calculation, to obtain scaled array of values


### logscaler
an extension of logarithm to whole real numbers for feature scaling purpose. log-scaler has one parameter to control the behaviour around zero