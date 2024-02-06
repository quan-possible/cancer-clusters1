"""
SUPPORT dataset.
Based on the code from Chapfuwa et al.:
    https://github.com/paidamoyo/survival_cluster_analysis
"""
import os

import numpy as np
import pandas
import pandas as pd


from sklearn.preprocessing import StandardScaler


def generate_data(seed=42):
    np.random.seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'support2.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=0)
    to_drop = ['hospdead', 'death', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'd.time', 'aps', 'sps', 'surv2m', 'surv6m',
               'totmcst']
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    # Preprocess
    one_hot_encoder_list = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'sfdm2']
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encoder_list)

    data_frame = log_transform(data_frame, transform_ls=['totmcst', 'totcst', 'charges', 'pafi', 'sod'])
    print("na columns:{}".format(data_frame.columns[data_frame.isnull().any()].tolist()))
    t_data = data_frame[['d.time']]
    e_data = data_frame[['death']]
    # dzgroup roughly corresponds to the diagnosis; more fine-grained than dzclass
    c_data = data_frame[['death']]
    c_data['death'] = c_data['death'].astype('category')
    c_data['death'] = c_data['death'].cat.codes

    x_data = data_frame.drop(labels=to_drop, axis=1)

    encoded_indices = one_hot_indices(x_data, one_hot_encoder_list)
    include_idx = set(np.array(sum(encoded_indices, [])))
    mask = np.array([(i in include_idx) for i in np.arange(x_data.shape[1])])
    print("head of x data:{}, data shape:{}".format(x_data.head(), x_data.shape))
    print("data description:{}".format(x_data.describe()))
    covariates = np.array(x_data.columns.values)
    print("columns:{}".format(covariates))
    x = np.array(x_data).reshape(x_data.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    c = np.array(c_data).reshape(len(c_data))

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    c = c[idx]

    # Normalization
    t = t / np.max(t) + 0.001
    scaler = StandardScaler()
    scaler.fit(x[:, ~mask])
    x[:, ~mask] = scaler.transform(x[:, ~mask])

    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))

    imputation_values = get_train_median_mode(x=x[train_idx], categorial=encoded_indices)

    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=imputation_values),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=imputation_values),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=imputation_values)
    }

    preprocessed['train']['c'] = c[train_idx]
    preprocessed['valid']['c'] = c[valid_idx]
    preprocessed['test']['c'] = c[test_idx]

    return preprocessed


def generate_support(seed=42):
    preproc = generate_data(seed)

    x_train = preproc['train']['x']
    x_valid = preproc['valid']['x']
    x_test = preproc['test']['x']

    t_train = preproc['train']['t']
    t_valid = preproc['valid']['t']
    t_test = preproc['test']['t']

    d_train = preproc['train']['e']
    d_valid = preproc['valid']['e']
    d_test = preproc['test']['e']

    c_train = preproc['train']['c']
    c_valid = preproc['valid']['c']
    c_test = preproc['test']['c']

    return x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test


def one_hot_encoder(data, encode):
    print("Encoding data:{}".format(data.shape))
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode, columns=encode)
    print("head of data:{}, data shape:{}".format(
        data_encoded.head(), data_encoded.shape))
    print("Encoded:{}, one_hot:{}{}".format(
        encode, encoded.shape, encoded[0:5]))
    return encoded


def log_transform(data, transform_ls):
    dataframe_update = data

    def transform(x):
        constant = 1e-8
        transformed_data = np.log(x + constant)
        # print("max:{}, min:{}".format(np.max(transformed_data), np.min(transformed_data)))
        return np.abs(transformed_data)

    for column in transform_ls:
        df_column = dataframe_update[column]
        print(" before log transform: column:{}{}".format(
            column, df_column.head()))
        print("stats:max: {}, min:{}".format(df_column.max(), df_column.min()))
        dataframe_update[column] = dataframe_update[column].apply(transform)
        print(" after log transform: column:{}{}".format(
            column, dataframe_update[column].head()))
    return dataframe_update


def formatted_data(x, t, e, idx, imputation_values=None):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])
    if imputation_values is not None:
        impute_covariates = impute_missing(
            data=covariates, imputation_values=imputation_values)
    else:
        impute_covariates = x
    survival_data = {'x': impute_covariates, 't': death_time, 'e': censoring}
    assert np.sum(np.isnan(impute_covariates)) == 0
    return survival_data


def get_train_median_mode(x, categorial):
    categorical_flat = flatten_nested(categorial)
    print("categorical_flat:{}".format(categorical_flat))
    imputation_values = []
    print("len covariates:{}, categorical:{}".format(
        x.shape[1], len(categorical_flat)))
    median = np.nanmedian(x, axis=0)
    mode = []
    for idx in np.arange(x.shape[1]):
        a = x[:, idx]
        (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_idx = a[index]
        mode.append(mode_idx)
    for i in np.arange(x.shape[1]):
        if i in categorical_flat:
            imputation_values.append(mode[i])
        else:
            imputation_values.append(median[i])
    print("imputation_values:{}".format(imputation_values))
    return imputation_values


def missing_proportion(dataset):
    missing = 0
    columns = np.array(dataset.columns.values)
    for column in columns:
        missing += dataset[column].isnull().sum()
    return 100 * (missing / (dataset.shape[0] * dataset.shape[1]))


def one_hot_indices(dataset, one_hot_encoder_list):
    indices_by_category = []
    for colunm in one_hot_encoder_list:
        values = dataset.filter(regex="{}_.*".format(colunm)).columns.values
        # print("values:{}".format(values, len(values)))
        indices_one_hot = []
        for value in values:
            indice = dataset.columns.get_loc(value)
            # print("column:{}, indice:{}".format(colunm, indice))
            indices_one_hot.append(indice)
        indices_by_category.append(indices_one_hot)
    # print("one_hot_indices:{}".format(indices_by_category))
    return indices_by_category


def flatten_nested(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened


def print_missing_prop(covariates):
    missing = np.array(np.isnan(covariates), dtype=float)
    shape = np.shape(covariates)
    proportion = np.sum(missing) / (shape[0] * shape[1])
    print("missing_proportion:{}".format(proportion))


def impute_missing(data, imputation_values):
    copy = data
    for i in np.arange(len(data)):
        row = data[i]
        indices = np.isnan(row)

        for idx in np.arange(len(indices)):
            if indices[idx]:
                # print("idx:{}, imputation_values:{}".format(idx, np.array(imputation_values)[idx]))
                copy[i][idx] = imputation_values[idx]
    # print("copy;{}".format(copy))
    return copy
