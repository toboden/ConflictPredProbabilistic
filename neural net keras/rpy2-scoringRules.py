"""
@author: Tim Janke, Energy Information Networks & Systems Lab @ TU Darmstadt, Germany

Mostly wrapper functions for the R scoringRules package using rpy2.
See https://cran.r-project.org/web/packages/scoringRules/ for details.
"""

import os
os.environ['R_HOME'] = 'C:/Users/Tobias/AppData/Local/Programs/R/R-42~1.1'


import numpy as np
import rpy2.robjects as R
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

# install scoring rules if necessary:
if rpackages.isinstalled("scoringRules") == False:
    r_utils = rpackages.importr('utils')
    r_utils.chooseCRANmirror(ind=1) # select the first mirror in the list
    r_utils.install_packages(R.vectors.StrVector(["scoringRules"])) #install scoringRules
scoringRules = rpackages.importr('scoringRules') # import scoringRules package
numpy2ri.activate()


# helper function to convert Python types to R types
def convert_to_Rtype(x):
    if x is None:
        x = R.NULL    
    elif type(x) is str:
        pass
    elif type(x) is int:
        pass
    elif type(x) is float:
        pass
    elif type(x) is np.ndarray:
        if x.ndim == 1:
            x = R.vectors.FloatVector(x)
        elif x.ndim == 2:
            x = R.r.matrix(x, nrow=x.shape[0], ncol=x.shape[1])
        else:
            raise TypeError("Input must be one of following: None, str, int, float, (n,) numpy array, (n,m) numpy array.")
    return x


########## univariate scores ###########
# Continuous Ranked Probability Score (CRPS)
def crps_sample(y, dat, w=None, return_single_scores=False):
    """
    Compute Continuous Ranked Probability Score (CRPS) from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape(n_examples,)
        True values.
    dat : array, shape (n_examples, n_samples)
        Predictive scenarios.
    w : array matching shape of dat, optional
        Array of weights. The default is None.

    Returns
    -------
    float or tuple of (float, array)
        Returns average CRPS.
        If return_single_scores is True also returns array of scores for single examples.

    """
    scores = scoringRules.crps_sample(y=convert_to_Rtype(y),
                                     dat=convert_to_Rtype(dat),
                                     method="edf",
                                     w=convert_to_Rtype(w))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)        


# Log Score
def logs_sample(y, dat, bw=None, return_single_scores=False):
    """
    Compute log-Score from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape(n_examples,)
        True values.
    dat : array, shape (n_examples, n_samples)
        Predictive scenarios.
    bw : array matching shape of y, optional
        Array of bandwiths for KDE. The default is None.

    Returns
    -------
    float or tuple of (float, array)
        Returns average log-Score.
        If return_single_scores is True, also returns array of scores for single examples.

    """
    scores = scoringRules.logs_sample(y=convert_to_Rtype(y),
                                     dat=convert_to_Rtype(dat),
                                     bw=convert_to_Rtype(bw))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)


########## multivariate scores ###########

# Energy Score
def es_sample(y, dat, return_single_scores=False):
    """
    Compute mean energy score from samples of the predictive distribution.

    Parameters
    ----------
    y : array, shape (n_examples, n_dim)
        True values.
    dat : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    return_single_scores : bool, optional
        Return score for single examples. The default is False.

    Returns
    -------
    float or tuple of (float, array)
        Mean energy score. If return_single_scores is True also returns scores for single examples.

    """
    assert y.shape[0] == dat.shape[0], "y and dat must contain same number of examples."
    assert y.shape[1] == dat.shape[1], "Examples in y and dat must have same dimension."

    scores = []
    for i in range(dat.shape[0]):
        scores.append(scoringRules.es_sample(y=convert_to_Rtype(y[i,:]), 
                                             dat=convert_to_Rtype(dat[i,:,:])))
    if return_single_scores:
        return np.mean(scores), np.asarray(scores)
    else:
        return np.mean(scores)
