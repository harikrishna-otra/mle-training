import argparse
import logging
from datetime import datetime
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np

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

exp_id = "Experiment6"
experiment_id = mlflow.create_experiment(exp_id)

with mlflow.start_run(
    run_name="PARENT_RUN", experiment_id=experiment_id
) as parent_run:
    mlflow.log_param("Parent", "yes")

    with mlflow.start_run(
        run_name="Data Ingestion", experiment_id=experiment_id, nested=True
    ) as child_run:
        mlflow.log_param("Data Ingestion", "yes")

        try:
            housing = Datapreparation.load_housing_data(args.load_raw_data)
            logging.info("Data is loaded")
        except Exception as e:
            logging.error("Error while loading data", e)

        mlflow.log_param("Load raw data", "yes")

        try:
            (
                housing_prepared,
                X_test_prepared,
                housing_labels,
                y_test,
            ) = Datapreparation.split_data(housing)
            logging.info("Data is spliting in train and test")
        except Exception as e:
            logging.error(
                "Error while spliting the data into train and test", e
            )

        mlflow.log_param("Split data", "yes")

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

        mlflow.log_param("Save data", "yes")

    param_grid = [
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
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
            "Score = {} and Parameters = {}".format(
                np.sqrt(-mean_score), params
            )
        )
        print(
            "Random Forest model (Bootstrap = {:f}, n_estimators = {:f}, max_features = {:f}):".format(  # noqa
                params["bootstrap"],
                params["n_estimators"],
                params["max_features"],
            )
        )
        print("Root Mean Square Error : {:f}".format(mean_score))

        with mlflow.start_run(
            run_name="Model Training", experiment_id=experiment_id, nested=True
        ):
            mlflow.log_param("Bootstrap", params["bootstrap"])
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_param("max_features", params["max_features"])
            mlflow.log_metric("Root Mean Square Error", mean_score)

    final_model = grid_search.best_estimator_
    logging.info("Final model = {}".format(final_model))

    try:
        model_training.save_model(final_model, args.save_model)
        logging.info("Model saved successfully")
    except Exception as e:
        logging.error("Error while saving the model", e)

    mlflow.log_artifact(args.save_model)
    print("Save to: {}".format(mlflow.get_artifact_uri()))

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    mlflow.sklearn.log_model(final_model, "model")

    with mlflow.start_run(
        run_name="Model Scoring", experiment_id=experiment_id, nested=True
    ) as child_run:
        mlflow.log_param("Model Scoring", "yes")

        try:
            final_predictions_data, final_rmse = score_model.score_test_data(
                X_test_prepared, y_test, final_model
            )
            logging.info("Test data is score, RMSE = {}".format(final_rmse))
        except Exception as e:
            logging.error("Error while saving the model", e)

        mlflow.log_metric("Root Mean Square Error for Test Set", mean_score)

        try:
            score_model.save_predictions(
                final_predictions_data, args.save_pred
            )
            logging.info("Predictions saved successfully")
        except Exception as e:
            logging.error("Error while saving the predictions", e)
