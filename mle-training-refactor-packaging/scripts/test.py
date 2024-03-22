import argparse
from src.house_price_prediction.ingest_data import Datapreparation

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

housing = Datapreparation.load_housing_data(args.load_raw_data)

print(housing)
