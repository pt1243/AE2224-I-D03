import pathlib
import glob

from scipy.io import loadmat
import numpy as np

from .paths import MAT_DIR, NPZ_DIR


def main():
    VERSION_FILE = NPZ_DIR.joinpath('version.txt')
    NPZ_VERSION = 1

    if not pathlib.Path.exists(NPZ_DIR):
        pathlib.Path.mkdir(NPZ_DIR)

    if pathlib.Path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            version_str = f.readlines()[0]
            try:
                found_version = int(version_str)
            except ValueError:
                found_version = 0
    else:
        found_version = 0

    process = True if found_version < NPZ_VERSION else False

    def mat_to_npz(input_file, output_file):
        contents = loadmat(input_file)

        wave = contents['wave'][0][0]   # Top-level MATLAB struct

        # info = wave[0]

        desc = wave[1]
        Ts = desc[0][0][0][0][0]
        fs = desc[0][0][1][0][0]
        desc_array = np.ascontiguousarray([Ts, fs])

        stack = np.hstack((wave[3], wave[2]))
        x = np.ascontiguousarray(stack[:, 0])  # times
        y = np.ascontiguousarray(stack[:, 1])  # voltage measurements

        np.savez_compressed(output_file, x=x, y=y, desc=desc_array)
        return

    def output_file(input_file):
        *_, cycle, filename = input_file.parts
        output_folder = NPZ_DIR.joinpath(cycle)
        output_filename = output_folder.joinpath(filename).with_suffix('.npz')
        return output_folder, output_filename

    if process:
        input_files = [MAT_DIR.joinpath(f) for f in glob.glob('**/*.mat', root_dir=MAT_DIR)]

        for file in input_files:
            output_folder, output_filename = output_file(file)
            if not pathlib.Path.exists(output_folder):
                pathlib.Path.mkdir(output_folder)

            mat_to_npz(file, output_filename)

        with open(VERSION_FILE, 'w+') as f:
            f.write(str(NPZ_VERSION))


if __name__ == '__main__':
    main()
