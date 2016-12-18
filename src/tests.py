from dataset_loader import DatasetLoader


for dataset in ['sentence_polarity']:#DatasetLoader.valid_datasets:
    data = DatasetLoader(dataset=dataset, vectorize='count', rep_mode='dense', positive_cat=1, feat_sel=0.1)
    print dataset + ' ' + str(data.num_features())
