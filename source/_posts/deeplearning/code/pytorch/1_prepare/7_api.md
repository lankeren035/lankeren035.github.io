---
title: 1.7 查阅文档
date: 2024-2-2 14:00:00
tags: [机器学习,pytorch]
categories: [机器学习]
comment: true
toc: true
---
#  
<!--more-->
# 7. 查阅文档
## 7.1 查找模块中的所有函数和类
- dir()


```python
import torch
print(dir(torch.distributions))
```

    ['AbsTransform', 'AffineTransform', 'Bernoulli', 'Beta', 'Binomial', 'CatTransform', 'Categorical', 'Cauchy', 'Chi2', 'ComposeTransform', 'ContinuousBernoulli', 'CorrCholeskyTransform', 'CumulativeDistributionTransform', 'Dirichlet', 'Distribution', 'ExpTransform', 'Exponential', 'ExponentialFamily', 'FisherSnedecor', 'Gamma', 'Geometric', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Independent', 'IndependentTransform', 'Kumaraswamy', 'LKJCholesky', 'Laplace', 'LogNormal', 'LogisticNormal', 'LowRankMultivariateNormal', 'LowerCholeskyTransform', 'MixtureSameFamily', 'Multinomial', 'MultivariateNormal', 'NegativeBinomial', 'Normal', 'OneHotCategorical', 'OneHotCategoricalStraightThrough', 'Pareto', 'Poisson', 'PowerTransform', 'RelaxedBernoulli', 'RelaxedOneHotCategorical', 'ReshapeTransform', 'SigmoidTransform', 'SoftmaxTransform', 'SoftplusTransform', 'StackTransform', 'StickBreakingTransform', 'StudentT', 'TanhTransform', 'Transform', 'TransformedDistribution', 'Uniform', 'VonMises', 'Weibull', 'Wishart', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'bernoulli', 'beta', 'biject_to', 'binomial', 'categorical', 'cauchy', 'chi2', 'constraint_registry', 'constraints', 'continuous_bernoulli', 'dirichlet', 'distribution', 'exp_family', 'exponential', 'fishersnedecor', 'gamma', 'geometric', 'gumbel', 'half_cauchy', 'half_normal', 'identity_transform', 'independent', 'kl', 'kl_divergence', 'kumaraswamy', 'laplace', 'lkj_cholesky', 'log_normal', 'logistic_normal', 'lowrank_multivariate_normal', 'mixture_same_family', 'multinomial', 'multivariate_normal', 'negative_binomial', 'normal', 'one_hot_categorical', 'pareto', 'poisson', 'register_kl', 'relaxed_bernoulli', 'relaxed_categorical', 'studentT', 'transform_to', 'transformed_distribution', 'transforms', 'uniform', 'utils', 'von_mises', 'weibull', 'wishart']
    

## 7.2 查找特定函数和类的使用
- help()


```python
help(torch.ones)
```

    Help on built-in function ones in module torch:
    
    ones(...)
        ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
        
        Returns a tensor filled with the scalar value `1`, with the shape defined
        by the variable argument :attr:`size`.
        
        Args:
            size (int...): a sequence of integers defining the shape of the output tensor.
                Can be a variable number of arguments or a collection like a list or tuple.
        
        Keyword arguments:
            out (Tensor, optional): the output tensor.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
            layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
                Default: ``torch.strided``.
            device (:class:`torch.device`, optional): the desired device of returned tensor.
                Default: if ``None``, uses the current device for the default tensor type
                (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU
                for CPU tensor types and the current CUDA device for CUDA tensor types.
            requires_grad (bool, optional): If autograd should record operations on the
                returned tensor. Default: ``False``.
        
        Example::
        
            >>> torch.ones(2, 3)
            tensor([[ 1.,  1.,  1.],
                    [ 1.,  1.,  1.]])
        
            >>> torch.ones(5)
            tensor([ 1.,  1.,  1.,  1.,  1.])
    
    


```python
list?
```

    [1;31mInit signature:[0m [0mlist[0m[1;33m([0m[0miterable[0m[1;33m=[0m[1;33m([0m[1;33m)[0m[1;33m,[0m [1;33m/[0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mDocstring:[0m     
    Built-in mutable sequence.
    
    If no argument is given, the constructor creates a new empty list.
    The argument must be an iterable if specified.
    [1;31mType:[0m           type
    [1;31mSubclasses:[0m     _HashedSeq, StackSummary, _Threads, ConvertingList, DeferredConfigList, _ymd, SList, ParamSpec, _ConcatenateGenericAlias, _ImmutableLineList, ...
