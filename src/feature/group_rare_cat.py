import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    '''
    A custom transformer that groups rare and unseen categories
    in categorical columns into a single category called 'other'.
    Rare categories are defined as those that appear
    in less than a specified proportion of the data.
    '''
    
    def __init__(self, columns=None, min_proportion=0.01):
        self.columns = columns
        self.min_proportion = min_proportion

    def fit(self, X, y=None):
        selected_columns = (
            list(self.columns)
            if self.columns is not None
            else list(
                X.select_dtypes(
                    include=['object', 'category']
                ).columns
            )
        )

        self.columns_to_group_ = []
        self.frequent_categories_ = {}

        for col in selected_columns:
            proportions = X[col].value_counts(normalize=True)

            rare_categories = proportions[
                proportions < self.min_proportion
            ].index

            frequent_categories = proportions[
                proportions >= self.min_proportion
            ].index

            if len(rare_categories) >= 2:
                self.columns_to_group_.append(col)
                self.frequent_categories_[col] = set(
                    frequent_categories
                )
            else:
                # Keep all known categories if grouping is skipped
                self.frequent_categories_[col] = set(
                    proportions.index
                )

        return self

    def transform(self, X):
        X_transformed = X.copy()

        for col in self.columns_to_group_:
            frequent_categories = self.frequent_categories_[col]

            X_transformed[col] = X_transformed[col].where(
                X_transformed[col].isin(frequent_categories),
                'other'
            )

        return X_transformed
    

