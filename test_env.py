import gresearch_crypto

result_file_name = 'submission.csv'

env = gresearch_crypto.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission

for group_id, (test_df, sample_prediction_df) in enumerate(iter_test):
    sample_prediction_df['Target'] = 0  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
    
    sample_prediction_df['group_num'] = group_id
    if group_id == 0:
        sample_prediction_df.to_csv(result_file_name, header=True, index=False, columns=['group_num', 'row_id', 'Target'])
    else:
        sample_prediction_df.to_csv(result_file_name, header=False, index=False, columns=['group_num', 'row_id', 'Target'])
        