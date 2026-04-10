from .base import AbstractDataset

import pickle
import pandas as pd


class _OursBaseDataset(AbstractDataset):
    dataset_code = None
    raw_filename = None

    @classmethod
    def code(cls):
        return cls.dataset_code

    @classmethod
    def url(cls):
        return None

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return [cls.raw_filename]

    def maybe_download_raw_dataset(self):
        pass

    def preprocess(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        self.maybe_download_raw_dataset()
        df = self.load_ratings_df()
        df = self.remove_immediate_repeats(df)
        df, umap, smap = self.densify_index(df)
        train, val, test = self.split_df(df, len(umap))
        dataset = {
            'train': train,
            'val': val,
            'test': test,
            'umap': umap,
            'smap': smap,
        }
        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath(self.raw_filename)
        df = pd.read_csv(file_path, sep='\t')
        expected = {'uid', 'sid', 'timestamp'}
        if not expected.issubset(set(df.columns)):
            raise ValueError(f'invalid file format: {file_path}, need columns {sorted(expected)}')
        return df[['uid', 'sid', 'timestamp']]


class BeautyOursDataset(_OursBaseDataset):
    dataset_code = 'beauty_ours'
    raw_filename = 'beauty_ours.tsv'


class ElectronicsOursDataset(_OursBaseDataset):
    dataset_code = 'electronics_ours'
    raw_filename = 'electronics_ours.tsv'


class ML100KOursDataset(_OursBaseDataset):
    dataset_code = 'ml100k_ours'
    raw_filename = 'ml100k_ours.tsv'


class BeautyLongOursDataset(_OursBaseDataset):
    dataset_code = 'beautylong_ours'
    raw_filename = 'beautylong_ours.tsv'


class SportsOursDataset(_OursBaseDataset):
    dataset_code = 'sports_ours'
    raw_filename = 'sports_ours.tsv'


class SportsLongOursDataset(_OursBaseDataset):
    dataset_code = 'sportslong_ours'
    raw_filename = 'sportslong_ours.tsv'


class ToysOursDataset(_OursBaseDataset):
    dataset_code = 'toys_ours'
    raw_filename = 'toys_ours.tsv'


class ToysLongOursDataset(_OursBaseDataset):
    dataset_code = 'toyslong_ours'
    raw_filename = 'toyslong_ours.tsv'


class Toys50KOursDataset(_OursBaseDataset):
    dataset_code = 'toys50k_ours'
    raw_filename = 'toys50k_ours.tsv'

