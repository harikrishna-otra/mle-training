#main 
import argparse
import logging
from datetime import datetime

import numpy as np
from scipy.stats import randint

from src.house_price_prediction.ingest_data import Datapreparation
from src.house_price_prediction.score import score_model
from src.house_price_prediction.train import model_training

parser = argparse.ArgumentParser()

parser.add_argument(
    "--load-raw-data",
    help="Please share the raw data path",
    default="datasets/housing/",
)
parser.add_argument(
    "--load-processed-data",
    help="Please share the processed data path",
    default="datasets/processed/",
)
parser.add_argument(
    "--save-model",
    help="Please share the path to save the model",
    default="artifacts/",
)
parser.add_argument(
    "--save-pred",
    help="Please share the output path",
    default="datasets/output/",
)
parser.add_argument(
    "--log-level",
    help="Please provide the level for log generation",
    default="DEBUG",
)
parser.add_argument(
    "--log-path",
    help="Please share the output path",
    default="datasets/output/",
)
parser.add_argument(
    "--no-console-log",
    help="Please specify if you want to write logs to the console",
    action="store_true",
)

args = parser.parse_args()

timestamp = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

if args.log_level:
    if args.log_path:
        logging.basicConfig(
            level=logging.getLevelName(args.log_level),
            filename=args.log_path
            + args.log_level
            + "_"
            + timestamp
            + "_"
            + ".log",
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.addLevelName(args.log_level))

if args.no_console_log:
    logger = logging.getLogger("my-logger")
    logger.propagate = False

try:
    housing = Datapreparation.load_housing_data(args.load_raw_data)
    logging.info("Data is loaded")
except Exception as e:
    logging.error("Error while loading data", e)


try:
    (
        housing_prepared,
        X_test_prepared,
        housing_labels,
        y_test,
    ) = Datapreparation.split_data(housing)
    logging.info("Data is spliting in train and test")
except Exception as e:
    logging.error("Error while spliting the data into train and test", e)


try:
    Datapreparation.save_data(
        args.load_processed_data,
        housing_prepared,
        X_test_prepared,
        housing_labels,
        y_test,  # noqa
    )
    logging.info("Processed data is saved")
except Exception as e:
    logging.error("Error while saving the processed data", e)

try:
    linear_reg_rmse, linear_reg_mae = model_training.train_linear_regression(
        housing_prepared, housing_labels
    )
    logging.info(
        "Linear regression fitted successfully. Train Data RMSE = {}. Train Data MAE = {}".format(  # noqa
            linear_reg_rmse, linear_reg_mae
        )
    )
except Exception as e:
    logging.error("Error while fitting linear regression", e)


try:
    decision_tree_rmse = model_training.train_decision_tree(
        housing_prepared, housing_labels
    )
    logging.info(
        "Decision Tree fitted successfully. Train Data RMSE = {}".format(
            decision_tree_rmse
        )
    )
except Exception as e:
    logging.error("Error while fitting Decision Tree", e)


param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

try:
    rnd_search = model_training.train_random_forest_ran_src(
        housing_prepared, housing_labels, param_distribs
    )
    logging.info("Random Forest fitted successfully using random search")
except Exception as e:
    logging.error("Error while fitting Random Forest", e)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    logging.info(
        "Score = {} and Parameters = {}".format(np.sqrt(-mean_score), params)
    )

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

try:
    grid_search = model_training.train_random_forest_grid_src(
        housing_prepared, housing_labels, param_grid
    )
    logging.info("Random Forest fitted successfully using grid search")
except Exception as e:
    logging.error("Error while fitting Random Forest", e)


grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    logging.info(
        "Score = {} and Parameters = {}".format(np.sqrt(-mean_score), params)
    )


feature_importances = grid_search.best_estimator_.feature_importances_
logging.info(
    "Feature importance = {}".format(
        sorted(
            zip(feature_importances, housing_prepared.columns), reverse=True
        )
    )
)

final_model = grid_search.best_estimator_
logging.info("Final model = {}".format(final_model))

try:
    model_training.save_model(final_model, args.save_model)
    logging.info("Model saved successfully")
except Exception as e:
    logging.error("Error while saving the model", e)

try:
    final_predictions_data, final_rmse = score_model.score_test_data(
        X_test_prepared, y_test, final_model
    )
    logging.info("Test data is score, RMSE = {}".format(final_rmse))
except Exception as e:
    logging.error("Error while saving the model", e)

try:
    score_model.save_predictions(final_predictions_data, args.save_pred)
    logging.info("Predictions saved successfully")
except Exception as e:
    logging.error("Error while saving the predictions", e)
