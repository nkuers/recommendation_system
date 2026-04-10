from .ml_1m import ML1MDataset
from .beauty import BeautyDataset
from .video import VideoDataset
from .sports import SportsDataset
from .steam import SteamDataset
from .xlong import XLongDataset
from .ours import BeautyOursDataset, ElectronicsOursDataset, ML100KOursDataset, BeautyLongOursDataset, SportsOursDataset, SportsLongOursDataset, ToysOursDataset, ToysLongOursDataset, Toys50KOursDataset


DATASETS = {
    ML1MDataset.code(): ML1MDataset,
    BeautyDataset.code(): BeautyDataset,
    VideoDataset.code(): VideoDataset,
    SportsDataset.code(): SportsDataset,
    SteamDataset.code(): SteamDataset,
    XLongDataset.code(): XLongDataset,
    BeautyOursDataset.code(): BeautyOursDataset,
    ElectronicsOursDataset.code(): ElectronicsOursDataset,
    ML100KOursDataset.code(): ML100KOursDataset,
    BeautyLongOursDataset.code(): BeautyLongOursDataset,
    SportsOursDataset.code(): SportsOursDataset,
    SportsLongOursDataset.code(): SportsLongOursDataset,
    ToysOursDataset.code(): ToysOursDataset,
    ToysLongOursDataset.code(): ToysLongOursDataset,
    Toys50KOursDataset.code(): Toys50KOursDataset,
}


def dataset_factory(args):
    dataset = DATASETS[args.dataset_code]
    return dataset(args)

