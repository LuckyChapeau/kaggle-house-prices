import warnings
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

# ------------------------------------------------------------------------------
class TargetEncoder(object):
    def __init__(
        self, smoothing=1, inflection=10, columns=None, log=False, as_pandas=False
    ):
        if smoothing is not None:
            if not (smoothing > 0):
                raise ValueError(
                    "The smoothing parameter needs to be either None or a float/integer > 0."
                )
        if columns is not None:
            if not (
                all([type(c) == str for c in columns])
                or all([type(c) == int for c in columns])
            ):
                raise ValueError(
                    "Columns must either all be strings (keys) or all be integer indexes."
                )
        self.columns = columns
        self.smoothing = smoothing
        self.inflection = inflection
        self.log = log
        self.as_pandas = as_pandas
        self.is_fit_ = False

    def get_params(self, deep=False):
        return {
            "smoothing": self.smoothing,
            "inflection": self.inflection,
            "log": self.log,
        }

    def set_params(self, **params):
        if not all(
            [key in ["inflection", "smoothing", "log"] for key in params.keys()]
        ):
            raise ValueError(
                "You can only set the parameters 'inflection', 'smoothing' and 'log'"
            )

        for param in params.keys():
            if param == "inflection":
                self.inflection = params["inflection"]
            elif param == "smoothing":
                self.smoothing = params["smoothing"]
            elif param == "log":
                self.log = params["log"]

    # Helper function
    def get_lambda_(self, size):
        lambda_ = 1 / (1 + np.exp(-((size - self.inflection) / self.smoothing)))
        return lambda_

    # Helper function
    def get_target_encodings_(self, x, y):
        grouped_y = y.groupby(x)
        means = grouped_y.mean()

        # Regularization
        if self.smoothing is not None:
            sizes = grouped_y.size()
            lambdas = sizes.apply(self.get_lambda_)
            regularized_means = (
                lambdas * means + (1 - lambdas) * self.overall_target_mean_
            )
            return regularized_means.to_dict()

        return means.to_dict()

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of observations.")

        # Cast to pandas DataFrame/Series
        self.X_pandas_ = X_pandas = y_pandas = False

        if type(X) == pd.core.frame.DataFrame:
            X = X.copy()
            X_pandas = True
            self.X_pandas_ = True
        else:
            if self.columns and type(self.columns[0]) == str:
                raise ValueError(
                    "You must provide a pandas DataFrame for X if you want to address columns with strings (keys)."
                )
            X = pd.DataFrame(X)

        if type(y) == pd.core.series.Series:
            y = y.copy()
            y_pandas = True
        else:
            y = pd.Series(y)

        # Check indexes and assure correpondence
        if X_pandas and y_pandas:
            if set(X.index) != set(y.index):
                raise ValueError("There are non-matching pandas indexes in X and y.")
            if X.index.has_duplicates or y.index.has_duplicates:
                raise ValueError("There are non-unique indexes in X and/or y.")
        elif X_pandas:
            y.index = X.index.copy()
        elif y_pandas:
            X.index = y.index.copy()

        # Store overall target mean
        self.overall_target_mean_ = y.mean()

        # Remember columns as integer indexes (in .columns_)
        if self.columns is None:
            if self.X_pandas_:
                X.columns = pd.RangeIndex(X.shape[1])
            self.columns_ = X.dtypes.loc[X.dtypes == "object"].index.tolist()
        elif type(self.columns[0]) == str:
            self.columns_ = [i for i, c in enumerate(X.columns) if c in self.columns]
            X.columns = pd.RangeIndex(X.shape[1])
        else:
            self.columns_ = self.columns
            if self.X_pandas_:
                X.columns = pd.RangeIndex(X.shape[1])

        # For each column get target encodings
        target_encodings = X.loc[:, self.columns_].apply(
            self.get_target_encodings_, axis=0, y=y
        )
        self.target_encodings_ = target_encodings.to_dict()

        # Done
        self.is_fit_ = True
        return self

    def transform(self, X):
        if not self.is_fit_:
            raise RuntimeError("Please fit data first.")

        if self.X_pandas_ and not type(X) == pd.core.frame.DataFrame:
            warnings.warn(
                "TargetEncoder was fit with pandas DataFrame but trying to transform numpy array."
            )

        # Cast X to pandas DataFrame
        if type(X) == pd.core.frame.DataFrame:
            X = X.copy()
        else:
            X = pd.DataFrame(X)

        for col in self.columns_:
            col_key = X.columns[col]  # since col is integer index
            col_target_encodings = self.target_encodings_[col]
            encoded_categories = col_target_encodings.keys()
            categories_to_encode = X[col_key].unique()
            missing_categories = list(
                set(categories_to_encode) - set(encoded_categories)
            )
            if len(missing_categories) > 0:
                # Update targets to include missing categories
                col_target_encodings.update(
                    {
                        category: self.overall_target_mean_
                        for category in missing_categories
                    }
                )
            X[col_key] = X[col_key].replace(col_target_encodings)
            if self.log:
                X[col_key] = np.log(X[col_key])

        if self.as_pandas:
            return X
        else:
            return X.values

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


# ------------------------------------------------------------------------------
class ModelStacker(object):
    def __init__(
        self,
        models,
        meta_model,
        n_splits=5,
        shuffle=False,
        random_state=None,
        passthrough=False,
        poly_degree=None,
    ):
        self.models = models
        self.used_models_ = [model[0] for model in models]  # use all by default
        self.meta_model = copy.deepcopy(meta_model)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.passthrough = passthrough
        self.poly_degree = poly_degree

        # Init the KFold object
        self.kfold_ = KFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def cv_refit_models_(self, X, y):
        # Get and store cv splits
        self.splits_ = tuple(self.kfold_.split(X))

        # Collect refitted models here
        self.refitted_models_ = {}

        # Refit models
        for model in self.models:
            model_name = model[0]
            refitted_models = []  # collector list for the *current* model
            for split in self.splits_:
                train_indexes = split[0]
                refitted_model = copy.deepcopy(model[1])
                refitted_model.fit(X.iloc[train_indexes, :], y.iloc[train_indexes])
                refitted_models.append(refitted_model)
            self.refitted_models_[model_name] = tuple(refitted_models)

    def get_train_meta_data_set_(self, X, model_names=None):
        # Init the data frame
        if self.passthrough:
            train_meta_data_set = X.copy()
        else:
            train_meta_data_set = pd.DataFrame(index=X.index)

        collected_preds = pd.DataFrame(index=X.index)

        # Option to only use certain models
        # Use all by default
        if not model_names:
            model_names = self.refitted_models_.keys()

        # Build the meta data set
        for model_name in model_names:
            model_preds = pd.Series(index=X.index, dtype=float)
            for i, split in enumerate(self.splits_):
                refitted_model = self.refitted_models_[model_name][i]
                test_indexes = split[1]
                test_preds = refitted_model.predict(X.iloc[test_indexes])
                model_preds.iloc[test_indexes] = test_preds
            collected_preds = pd.concat([collected_preds, model_preds], axis=1)

        if self.poly_degree:
            self.poly_features_ = PolynomialFeatures(
                degree=self.poly_degree, include_bias=False, interaction_only=False
            )
            collected_preds = self.poly_features_.fit_transform(collected_preds)
            collected_preds = pd.DataFrame(collected_preds, index=X.index)

        collected_preds_col_names = [
            str(i) + "_model_preds" for i in range(collected_preds.shape[1])
        ]
        collected_preds.columns = collected_preds_col_names

        train_meta_data_set = pd.concat([train_meta_data_set, collected_preds], axis=1)

        self.train_meta_data_set_ = train_meta_data_set

    def get_prediction_meta_data_set_(self, X):
        # Init the data frame
        if self.passthrough:
            prediction_meta_data_set = X.copy()
        else:
            prediction_meta_data_set = pd.DataFrame(index=X.index)

        collected_preds = pd.DataFrame(index=X.index)

        # Use the average of the predictions of the refitted models
        for used_model in self.used_models_:
            model_preds_sum = pd.Series(0.0, index=X.index)
            for refitted_model in self.refitted_models_[used_model]:
                model_preds = refitted_model.predict(X)
                model_preds_sum += model_preds
            model_preds_mean = model_preds_sum / self.kfold_.get_n_splits()
            collected_preds = pd.concat([collected_preds, model_preds_mean], axis=1)

        if self.poly_degree:
            collected_preds = self.poly_features_.transform(collected_preds)
            collected_preds = pd.DataFrame(collected_preds, index=X.index)

        collected_preds_col_names = [
            str(i) + "_model_preds" for i in range(collected_preds.shape[1])
        ]
        collected_preds.columns = collected_preds_col_names

        prediction_meta_data_set = pd.concat(
            [prediction_meta_data_set, collected_preds], axis=1
        )

        return prediction_meta_data_set

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of observations.")

        # Cast to input data to pandas DataFrame/Series
        X_pandas = y_pandas = False

        if type(X) == pd.core.frame.DataFrame:
            X = X.copy()
            X_pandas = True
        else:
            X = pd.DataFrame(X)

        if type(y) == pd.core.series.Series:
            y = y.copy()
            y_pandas = True
        else:
            y = pd.Series(y)

        # Check indexes and assure correpondence
        if X_pandas and y_pandas:
            if set(X.index) != set(y.index):
                raise ValueError("There are non-matching pandas indexes in X and y.")
            if X.index.has_duplicates or y.index.has_duplicates:
                raise ValueError("There are non-unique indexes in X and/or y.")
        elif X_pandas:
            y.index = X.index.copy()
        elif y_pandas:
            X.index = y.index.copy()

        # Get the refitted models
        self.cv_refit_models_(X, y)

        # Build the meta data set
        self.get_train_meta_data_set_(X)

        # Fit the meta model on the meta data
        self.meta_model.fit(self.train_meta_data_set_, y)

        # Done
        self.is_fit_ = True
        return self

    def predict(self, X):
        if not self.is_fit_:
            raise RuntimeError("Please fit data first.")

        # Cast X to pandas DataFrame
        if type(X) == pd.core.frame.DataFrame:
            X = X.copy()
        else:
            X = pd.DataFrame(X)

        # Get the prediction meta data set
        prediction_meta_data_set = self.get_prediction_meta_data_set_(X)

        # Make the predictions on this data set using the fitted meta model
        preds = self.meta_model.predict(prediction_meta_data_set)

        return preds
