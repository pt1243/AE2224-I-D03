import pathlib


ROOT_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR: pathlib.Path = ROOT_DIR.joinpath('data')
MAT_DIR: pathlib.Path = DATA_DIR.joinpath('matlab_files')
RAW_DATA_DIR: pathlib.Path = DATA_DIR.joinpath('L2_S2')
NPZ_DIR: pathlib.Path = DATA_DIR.joinpath('npz_files')
