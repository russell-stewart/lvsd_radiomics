import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    `FeatureSelector`
    This implementation of an sklearn transformer allows our pipeline to:
        - Select some minimum number of features using RFECV
        - Drop features above some maximum threshold using heirarchical clustering
    Attributes:
    estimator: The base estimator used for RFECV
    min_features_to_select: Select at least this many most important features to carry out from the transformer...
    max_features_to_select: ...and no more than this many
    cross_correlation_threshold: For use in RFECV
    step: For use in RFECV
    cv: Either an int number of cross-validation splits, or an instantiated sklearn cross-validation splitter
    selector: Will be trained as the RFECV instance upon running `fit`
    n_jobs: Number of CPU jobs to allocate to RFECV
    scoring: Scoring metric to rank classifier performance in RFECV
    ranking_: Once determined in `fit`, this vector will contain feature rankings in an array parallel to the columns of $X$.
    """
    def __init__(self, estimator=None, min_features_to_select=1, max_features_to_select=50, step=1, cv=5 , scoring='f1' , cross_correlation_threshold=0.8 , n_jobs = -1):
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.max_features_to_select = max_features_to_select
        self.cross_correlation_threshold = cross_correlation_threshold
        self.step = step
        self.cv = cv
        self.selector = None
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.ranking_ = None

    def fit(self, X, y):
        """
        `fit`
        Determines which features meet importance criteria specified in init thorugh RFECV, plus heirarchical clustering if needed, given a training data set
        $X$ with ground-truth labels $y$
        """
        self.selector = RFECV(estimator=self.estimator,
                              step=self.step,
                              cv=self.cv,
                              scoring=self.scoring,
                              min_features_to_select=1,
                              n_jobs=self.n_jobs)#allow for minimum RANK size of 1, distinct from n features we'll choose
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")#a bunch of our rfecv logistic regressions won't converge; that's expected. shh.
            self.selector.fit(X, y)
        # If the number of selected features is less than the minimum, adjust
        if np.sum(self.selector.support_) < self.min_features_to_select:
            # Select the top 'min_features_to_select' features based on importance
            selected_features = np.argsort(self.selector.ranking_)[:self.min_features_to_select]
            self.selector.support_ = np.zeros(X.shape[1], dtype=bool)
            self.selector.support_[selected_features] = True
        elif np.sum(self.selector.support_) > self.max_features_to_select:
            self._cross_correlate_drop_features(X)
        self.ranking_ = self.selector.ranking_
        return self

    def _cross_correlate_drop_features(self , X):
        """
        `_cross_correlate_drop_features`
        Only used if RFECV returns more than `self.max_features_to_select`. Uses heirarchical clusltering by Pearson's coefficient
        to narrow feature list, chosing the most representative feature (by variance) from each cluster.
        """
        prior_mask_indices = np.where(self.selector.support_)[0]
        correlations = pd.DataFrame(X[:,self.selector.support_]).corr().values
        assert np.isnan(correlations).sum() == 0 , correlations
        dissimilarity = 1 - correlations
        Z = linkage(squareform(dissimilarity) , 'complete')
        labels = fcluster(Z , self.cross_correlation_threshold , criterion = 'distance')
        
        clustered = pd.DataFrame(correlations)
        clustered['sort_index'] = labels
        clustered.sort_values(by = 'sort_index' , inplace = True)
        clustered.drop('sort_index' , axis = 1 , inplace = True)
        clustered = clustered.T
        clustered['sort_index'] = labels
        clustered.sort_values(by = 'sort_index' , inplace = True)
        clustered.drop('sort_index' , axis = 1 , inplace = True)
        
        variance_by_cluster = pd.DataFrame({'Feature':clustered.index,'ClusterOrdinal':labels}).sort_values(
            by='ClusterOrdinal').groupby(
                by = 'ClusterOrdinal').apply(
                    lambda sub_c: clustered.loc[sub_c.Feature , sub_c.Feature].apply(
                        np.var , axis = 1)).reset_index().rename(
                            columns={'level_1':'Feature',0:'MeanIntraclusterVariance'}).sort_values(
                                by=['ClusterOrdinal','MeanIntraclusterVariance'])
        
        indices_in_prior_mask_indices = variance_by_cluster.drop_duplicates(subset = ['ClusterOrdinal'] , keep = 'first').Feature.tolist()
        new_mask_indices = prior_mask_indices[indices_in_prior_mask_indices]
        self.selector.support_ = np.zeros(X.shape[1], dtype=bool)
        self.selector.support_[new_mask_indices] = True
    
    def transform(self, X):
        """
        `transform`
        Passes only the features deemed important from $X$ forward.
        """
        return X[:, self.selector.support_]

    def get_support(self):
        """
        `get_support`
        Returns a binary mask of which columns from the training $X$ were ranked as important during `fit`
        """
        return self.selector.support_