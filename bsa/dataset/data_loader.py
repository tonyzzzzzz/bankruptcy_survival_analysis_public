import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def load_raw_data(filename):

  raw_data_by_gvkey_fyear = pd.read_csv(filename, index_col=['gvkey', 'fyear'])
  sorted_data_raw_data = raw_data_by_gvkey_fyear.sort_index()

  feature_columns = [col for col in sorted_data_raw_data.columns if col != 'IBankrupt']

  return sorted_data_raw_data[feature_columns], sorted_data_raw_data['IBankrupt']

def generate_labels(raw_labels):
    def calculate_te(company_data):
        # Change fyear index to columns
        fyears_value = company_data.index.get_level_values('fyear')

        # Calculate all bankrupted years in NumPy arrays
        bankrupted_years = company_data.loc[company_data > 0].index.get_level_values('fyear')
        bankrupted_years_np = bankrupted_years.values.flatten()

        # Calculate all missing years in NumPy arrays
        presented_years = fyears_value.values
        all_years = np.arange(start=np.min(presented_years), stop=(np.max(presented_years)+2))
        last_present_year_np = np.setdiff1d(all_years, presented_years) - 1

        # For definition 1, we treat both bankrupted years and missing years as event years
        event_years = np.concatenate([bankrupted_years_np, last_present_year_np])

        # Return T: Time to the earliest event year(union of bankrupted year and missing year)
        def calculate_t(x):
            dist = x - event_years
            dist = dist[dist <= 0] # Future only
            dist = np.abs(dist) # Take absolute distance
            return np.min(dist) + 1
        
        # Return T for Definition 1: T = nearest future bankruptcy year, if not present, then nearest future missing year
        # def calculate_t(x):
        #     dist = x - bankrupted_years_np
        #     dist = dist[dist <= 0] # Future only
        #     dist = np.abs(dist)
        #     if dist.size == 0:
        #         return presented_years.max() - x + 1
        #     return np.min(dist)
    
        # Return E: if there are any future bankrupted year, return 1, else return 0
        def calculate_e(x):
            dist = x - bankrupted_years_np
            dist = dist[dist <= 0]
            if dist.size == 0:
                return 0
            return 1

        # Company_data is Series object, change to DataFrame to add columns
        labels = company_data.to_frame()
        labels['T'] = fyears_value.map(calculate_t)
        labels['E'] = fyears_value.map(calculate_e)

        # Rename columns
        # labels = labels.set_axis(['IBankrupt', 'T', 'E'], axis=1, copy=False)
        labels = labels.set_axis(['IBankrupt', 'T', 'E'], axis=1)

        return labels
    
    # Calculate T, E columns for each company
    y = raw_labels.copy()
    y = y.groupby(level=0, group_keys=False).apply(calculate_te)

    return y

def drop_unknown_horizon(y, horizon : int):
    # For definition 2: 
    # Mask for all rows where bankrupted
    bankrupted_filter = y['E'] == 1
    # Mask for all rows where has information within horizon
    greater_than_horizon_filter = y['T'] > horizon
    # Use OR operation for two masks
    mask = bankrupted_filter | greater_than_horizon_filter

    y = y[mask]

    # Add column for companies bankrupted within horizon
    y['HBankrupt'] = ((y['T'] <= (horizon + 1)) & bankrupted_filter).astype(int)

    # For definition 1:
    # greater_than_horizon_filter = y['T'] > horizon
    # mask = greater_than_horizon_filter

    return y

# Standardization & Rolling Mean Std & TE Labels
def preprocess_data(x : pd.DataFrame, y):

    ## Binary rolling mean std

    ### PREPROCESS X ###
    x = x.copy()

    non_binary_columns = x.filter(regex="^(?!I[A-Z])", axis=1).columns.to_list()
    binary_columns = [i for i in x.columns.to_list() if i not in non_binary_columns]
    non_binary_feats = x[non_binary_columns]
    non_binary_groups = non_binary_feats.groupby("gvkey", group_keys=False)

    # # Rolling Mean Std
    rolling_mean : pd.DataFrame = non_binary_groups.apply(lambda x: x.rolling(window=5, min_periods=1).mean()) # Consider censored years
    rolling_mean = rolling_mean.add_suffix("_rolling_mean")
    rolling_std : pd.DataFrame = non_binary_groups.apply(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)
    rolling_std = rolling_std.add_suffix("_rolling_std")

    # Concatenate x
    non_binary_feats = pd.concat([non_binary_feats, rolling_mean, rolling_std], axis=1)

    # Normalize
    preprocessor = StandardScaler()
    non_binary_feats = pd.DataFrame(preprocessor.fit_transform(non_binary_feats), index=non_binary_feats.index, columns=non_binary_feats.columns)

    x = pd.concat([non_binary_feats, x[binary_columns]], axis=1)

    ### PREPROCESS Y ###
    outcomes = generate_labels(y)

    return x, outcomes

def splitting_function(features, labels, test_size=0.25):
  gv_keys = np.unique(labels.index.get_level_values(0).values)
  gv_keys_len = len(gv_keys)
  test_len = math.floor(gv_keys_len * test_size)
  train_len = gv_keys_len - test_len

  train_gv_keys = np.random.choice(gv_keys, train_len, replace=False)
  test_gv_keys = np.setdiff1d(gv_keys, train_gv_keys, assume_unique=True)

  return features.loc[train_gv_keys], features.loc[test_gv_keys], labels.loc[train_gv_keys], labels.loc[test_gv_keys] 