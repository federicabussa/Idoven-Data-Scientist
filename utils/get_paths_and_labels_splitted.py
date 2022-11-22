import numpy as np
import os
import pandas as pd

def get_paths_and_labels():

    # dataset = pd.read_csv('data/ptb-xl/labels-5classes.csv')
    dataset = pd.read_csv('data/ptb-xl/ptbxl_database.csv')
    statements = pd.read_csv('data/ptb-xl/scp_statements.csv', index_col = 0)

    dataset['label'] = dataset.scp_codes.apply(lambda x: x[1:-1].split(', '))
    dataset['label'] = dataset.label.apply(lambda label: [x[1:].split("': ") for x in label])
    dataset['label'] = dataset.label.apply(lambda label: [(x[0], float(x[1])) for x in label])
    dataset['likelihood'] = np.nan
    dataset['flag_multiclass'] = np.nan

    for i, label in dataset['label'].items():
        # find maximum likelihood
        max_likelihood = max([x[1] for x in label])
        # select scp codes corresponding to max. likelihood
        scp = [x[0] for x in label if x[1] == max_likelihood]
        # select correspondent diagnostic class
        diagnostic_classes = [statements.loc[scp_, 'diagnostic_class'] for scp_ in scp]
        # filter nan
        diagnostic_classes = [x for x in diagnostic_classes if not pd.isnull(x)]
        # remove duplicates
        diagnostic_classes = list(set(diagnostic_classes))
        # final classification
        dataset.at[i, 'label'] = ', '.join(diagnostic_classes)
        dataset.at[i, 'likelihood'] = max_likelihood
        if len(diagnostic_classes) == 1:
            dataset.at[i, 'flag_multiclass'] = 0
        else:
            dataset.at[i, 'flag_multiclass'] = 1

    dataset['flag_multiclass'] = dataset['flag_multiclass'].astype('int')

    # dataset = dataset[(dataset.flag_multiclass == 0) & (dataset.likelihood == 100)]
    dataset = dataset[(dataset.flag_multiclass == 0) & (dataset.likelihood == 100) & (pd.isna(dataset.static_noise))& (pd.isna(dataset.burst_noise))& (pd.isna(dataset.electrodes_problems)) ]

    dataset.filename_lr = dataset.filename_lr.apply(lambda x: x.replace('records100', 'records100_filt'))
    TRAINING_SET_FILE_PATHS = [os.path.join('data/ptb-xl', x+'.npy') for x in dataset[dataset.strat_fold < 9].filename_lr.tolist()]
    VALIDATION_SET_FILE_PATHS = [os.path.join('data/ptb-xl', x+'.npy') for x in dataset[dataset.strat_fold == 9].filename_lr.tolist()]
    TEST_SET_FILE_PATHS = [os.path.join('data/ptb-xl', x+'.npy') for x in dataset[dataset.strat_fold == 10].filename_lr.tolist()]

    labels_mapping = {k:i for i, k in enumerate(dataset.label.unique())}
    dataset.label = dataset.label.apply(lambda x: labels_mapping[x])

    TRAINING_SET_LABELS = dataset[dataset.strat_fold < 9]['label'].tolist()
    VALIDATION_SET_LABELS = dataset[dataset.strat_fold == 9]['label'].tolist()
    TEST_SET_LABELS = dataset[dataset.strat_fold == 10]['label'].tolist()

    assert len(TRAINING_SET_FILE_PATHS) == len(TRAINING_SET_LABELS)
    assert len(VALIDATION_SET_FILE_PATHS) == len(VALIDATION_SET_LABELS)
    assert len(TEST_SET_FILE_PATHS) == len(TEST_SET_LABELS)

    return TRAINING_SET_FILE_PATHS, TRAINING_SET_LABELS, VALIDATION_SET_FILE_PATHS, VALIDATION_SET_LABELS, TEST_SET_FILE_PATHS, TEST_SET_LABELS

