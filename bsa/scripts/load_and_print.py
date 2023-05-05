import sys

sys.path.append("dataset")

from data_loader import load_raw_data, drop_unknown_horizon, splitting_function

DATASET = "/Users/tonyzou/bankruptcy-survival-analysis/data/data_for_bankruptcy_prediction_no_lags_corrected.csv"

if __name__ == '__main__':
    raw_data, raw_label = load_raw_data(DATASET)
    y = drop_unknown_horizon(raw_label, horizon=3)
    x = raw_data.loc[y.index]
    print("Data and Label for Horizon 3")
    print(x)
    print(y)

    train_x, test_x, train_y, test_y = splitting_function(x, y)
