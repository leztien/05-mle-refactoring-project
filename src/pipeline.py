
import numpy as np, pandas as pd
from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline



class AbstractTransformer(ABC):
    """
    This (abstract) transformer is to be inherited by the other transformers.
    It may not be instantiated on its own.
    Implements 'fit' method and
    checks wether 'X' is a DataFrame
    """
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("The input 'X' must be a pandas.DataFrame")
        return self



### Data Cleaning and Imputation Transformers ###

class RowDropper(BaseEstimator, TransformerMixin, AbstractTransformer):
    def transform(self, X, y=None):
        # put the indeces of the rows to drop into this list
        idx = [15856,]
        X.drop(idx, axis=0, inplace=True)
        return X


class Imputer(BaseEstimator, TransformerMixin, AbstractTransformer):
    def transform(self, X, y=None):
        impute_columns = ['view', 'waterfront']
        for column in impute_columns:
            X[column].fillna(0, inplace=True)
        return X



### Feature Engineering Transformers ###

class SqftBasementCalculator(BaseEstimator, TransformerMixin, AbstractTransformer):
    def transform(self, X, y=None):
        X.eval('sqft_basement = sqft_living - sqft_above', inplace=True)
        X['sqft_basement'] = X['sqft_basement'].astype(float)
        assert X['sqft_basement'].isnull().sum() == 0, "'sqft_basement' must not contain any nan's at this point"
        return X


class LastKnownChangeFeature(BaseEstimator, TransformerMixin, AbstractTransformer):
    """See the explanation for this feature in the EDA notebook"""
    def transform(self, X, y=None):
        last_known_change = []

        for idx, yr_re in X.yr_renovated.items():
            if str(yr_re) == 'nan' or yr_re == 0.0:
                last_known_change.append(X.yr_built[idx])
            else:
                last_known_change.append(int(yr_re))

        X['last_known_change'] = last_known_change
        X.drop("yr_renovated", axis=1, inplace=True)
        X.drop("yr_built", axis=1, inplace=True)
        return X


class CenterDistanceFeature(BaseEstimator, TransformerMixin, AbstractTransformer):
    """See the explanation for this feature in the EDA notebook"""
    def transform(self, X, y=None):
        delta_lat = np.absolute(47.62774 - X['lat'])
        delta_long = np.absolute(-122.24194 - X['long'])
        X['center_distance']= ((delta_long* np.cos(np.radians(47.6219)))**2 
                                        + delta_lat**2)**(1/2)*2*np.pi*6378/360
        return X


class WaterDistanceFeature(BaseEstimator, TransformerMixin, AbstractTransformer):
    """See the explanation for this feature in the EDA notebook"""
    @staticmethod
    def _dist(long, lat, ref_long, ref_lat):
        delta_long = long - ref_long
        delta_lat = lat - ref_lat
        delta_long_corr = delta_long * np.cos(np.radians(ref_lat))
        return ((delta_long_corr)**2 +(delta_lat)**2)**(1/2)*2*np.pi*6378/360

    def transform(self, X, y=None):
        water_list= X.query('waterfront == 1')
        water_distance = []

        for idx, lat in X.lat.items():
            ref_list = []
            for x,y in zip(list(water_list.long), list(water_list.lat)):
                ref_list.append(self._dist(X.long[idx], X.lat[idx],x,y).min())
            water_distance.append(min(ref_list))
        
        X['water_distance'] = water_distance
        return X



feature_engineering_pipeline = make_pipeline(
    SqftBasementCalculator(),
    LastKnownChangeFeature(),
    CenterDistanceFeature(),
    WaterDistanceFeature()
)

preprocessing_pipeline = Pipeline([
    ('cleaning_pipeline', RowDropper()), # placeholder for a larger data cleaning pipeline (if need be)
    ('imputing_pipeline', Imputer()),
    ('feature_engineering_pipeline', feature_engineering_pipeline),
])

