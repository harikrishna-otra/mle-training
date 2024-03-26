import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class score_model:
    def score_test_data(X_test_prepared, y_test, final_model):
        final_predictions = final_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)

        final_predictions_data = pd.DataFrame(
            final_predictions, columns=["prediction"]
        )

        return final_predictions_data, final_rmse

    def save_predictions(final_predictions_data, path):
        final_predictions_data.to_csv(
            path + "scored_test_data.csv", index=False
        )
