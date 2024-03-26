import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


class model_training:
    def train_linear_regression(housing_prepared, housing_labels):
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)

        lin_mae = mean_absolute_error(housing_labels, housing_predictions)

        return lin_rmse, lin_mae

    def train_decision_tree(housing_prepared, housing_labels):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)

        return tree_rmse

    def train_random_forest_ran_src(
        housing_prepared, housing_labels, param_distribs
    ):
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(
            forest_reg,
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        rnd_search.fit(housing_prepared, housing_labels)

        return rnd_search

    def train_random_forest_grid_src(
        housing_prepared, housing_labels, param_grid
    ):
        forest_reg = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(
            forest_reg,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            return_train_score=True,
        )

        grid_search.fit(housing_prepared, housing_labels)

        return grid_search

    def save_model(final_model, path):
        pickle.dump(
            final_model,
            open(
                path + "final_model.sav",
                "wb",
            ),
        )
