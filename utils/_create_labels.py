import pandas as pd
import numpy as np
import os
import wfdb

dataset = pd.read_csv('data/ptb-xl/ptbxl_database.csv')
statements = pd.read_csv('data/ptb-xl/scp_statements.csv', index_col = 0)

norm_scp = statements[statements.diagnostic_class == 'NORM'].index.tolist()
sttc_scp = statements[statements.diagnostic_class == 'STTC'].index.tolist()
mi_scp = statements[statements.diagnostic_class == 'MI'].index.tolist()
cd_scp = statements[statements.diagnostic_class == 'CD'].index.tolist()
hyp_scp = statements[statements.diagnostic_class == 'HYP'].index.tolist()

# TRAINING_SET_FILE_PATHS = [os.path.join('data/ptb-xl', x) for x in dataset[dataset.strat_fold < 9].filename_lr.tolist()]
# VALIDATION_SET_FILE_PATHS = [os.path.join('data/ptb-xl', x) for x in dataset[dataset.strat_fold == 9].filename_lr.tolist()]
# TEST_SET_FILE_PATHS = [os.path.join('data/ptb-xl', x) for x in dataset[dataset.strat_fold == 10].filename_lr.tolist()]


dataset['label'] = dataset.scp_codes.apply(lambda x: x[1:-1].split(', '))
dataset['label'] = dataset.label.apply(lambda label: [x[1:].split("': ") for x in label])
dataset['label'] = dataset.label.apply(lambda label: [(x[0], float(x[1])) for x in label])
dataset['likelihood'] = np.nan
dataset['flag_multiclass'] = np.nan

dataset['label'].apply(lambda diagnosis: 1 if any(diag in norm_scp for diag, _ in diagnosis) else 0).sum() # 9514 instead od 9517
dataset['label'].apply(lambda diagnosis: 1 if any(diag in sttc_scp for diag, _ in diagnosis) else 0).sum() # 5235 instead of 5237
dataset['label'].apply(lambda diagnosis: 1 if any(diag in mi_scp for diag, _ in diagnosis) else 0).sum() # 5469 instead of 5473
dataset['label'].apply(lambda diagnosis: 1 if any(diag in cd_scp for diag, _ in diagnosis) else 0).sum() # 4898 instead of 4901
dataset['label'].apply(lambda diagnosis: 1 if any(diag in hyp_scp for diag, _ in diagnosis) else 0).sum() # 2649 - correct 


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

dataset[(dataset.flag_multiclass == 0) & (dataset.likelihood == 100)]

dataset[(dataset.flag_multiclass == 0) & (dataset.likelihood == 100) & (pd.isna(dataset.static_noise))& (pd.isna(dataset.burst_noise))& (pd.isna(dataset.electrodes_problems)) ]

# dataset.to_csv('data/ptb-xl/labels-5classes.csv', index=False)

