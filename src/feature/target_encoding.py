
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GeoLevel3TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Encode geo_level_3_id using damage_grade proportions.

    If a geo_level_3_id appears fewer than min_count times,
    use its parent geo_level_2_id proportions instead.
    Unseen parent IDs fall back to global proportions.
    """

    def __init__(self, min_count=10):
        self.min_count = min_count

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)

        # Target classes and global proportions
        self.classes_ = np.sort(y.dropna().unique())

        self.global_proportions_ = (
            y.value_counts(normalize=True)
            .reindex(self.classes_, fill_value=0)
        )

        # Number of training samples for each geo_level_3_id
        self.geo3_counts_ = X['geo_level_3_id'].value_counts()

        # Target proportions for each geo_level_3_id
        self.geo3_mapping_ = pd.crosstab(
            X['geo_level_3_id'],
            y,
            normalize='index'
        ).reindex(columns=self.classes_, fill_value=0)

        # Target proportions for each parent geo_level_2_id
        self.geo2_mapping_ = pd.crosstab(
            X['geo_level_2_id'],
            y,
            normalize='index'
        ).reindex(columns=self.classes_, fill_value=0)

        return self

    def transform(self, X):
        # Encode geo_level_3_id
        geo3_encoded = self.geo3_mapping_.reindex(
            X['geo_level_3_id'].to_numpy()
        )
        geo3_encoded.index = X.index

        # Identify rare or unseen geo_level_3_id values
        counts = X['geo_level_3_id'].map(self.geo3_counts_)
        rare_mask = counts.lt(self.min_count) | counts.isna()

        # Get parent geo_level_2_id proportions
        geo2_encoded = self.geo2_mapping_.reindex(
            X['geo_level_2_id'].to_numpy()
        )
        geo2_encoded.index = X.index

        # Use global proportions if the parent ID is also unseen
        geo2_encoded = geo2_encoded.fillna(self.global_proportions_)

        # Replace rare or unseen geo_level_3_id encodings
        geo3_encoded.loc[rare_mask] = geo2_encoded.loc[rare_mask]

        # Give the encoded columns clear names
        geo3_encoded.columns = [
            f"geo_level_3_target_{cls}"
            for cls in self.classes_
        ]

        # Replace the original geo_level_3_id column
        X_out = X.drop(columns=['geo_level_3_id', 'geo_level_2_id']).copy()

        return pd.concat([X_out, geo3_encoded], axis=1)
