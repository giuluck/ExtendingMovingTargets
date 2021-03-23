from tensorflow.keras.utils import Sequence


class MLPBatchGenerator(Sequence):
    def __init__(self, data, batch_size):
        super(MLPBatchGenerator).__init__()
        data = data.reset_index(drop=True)
        data['avg_rating'] = (data['avg_rating'] - data['avg_rating'].mean()) / data['avg_rating'].std()
        data['num_reviews'] = (data['num_reviews'] - data['num_reviews'].mean()) / data['num_reviews'].std()
        data['D'] = (data['dollar_rating'] == 'D').astype('float')
        data['DD'] = (data['dollar_rating'] == 'DD').astype('float')
        data['DDD'] = (data['dollar_rating'] == 'DDD').astype('float')
        data['DDDD'] = (data['dollar_rating'] == 'DDDD').astype('float')
        data['batch'] = data.index // batch_size
        self.batches = [b.drop(['batch', 'dollar_rating'], axis=1).reset_index(drop=True) for _, b in
                        data.groupby('batch')]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]
        x = batch[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        y = batch['clicked']
        return x.values, y.values


class SBRBatchGenerator(MLPBatchGenerator):
    def __init__(self, data, batch_size):
        super(SBRBatchGenerator, self).__init__(
            data.sort_values(['index', 'clicked'], ascending=[True, False]).reset_index(drop=True),
            len(data[data['index'] == 0]) * batch_size
        )
        self.batches = [b.drop('index', axis=1) for b in self.batches]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]
        x = batch[['avg_rating', 'num_reviews', 'D', 'DD', 'DDD', 'DDDD']]
        y = batch[['clicked', 'monotonicity']]
        return x.values, y.values
