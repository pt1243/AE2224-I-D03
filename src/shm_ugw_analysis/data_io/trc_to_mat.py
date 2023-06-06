import glob
import pathlib

import matlab.engine

from .paths import MAT_DIR, RAW_DATA_DIR


def main():
    VERSION_FILE = MAT_DIR.joinpath('version.txt')
    MAT_VERSION = 1

    if not pathlib.Path.exists(MAT_DIR):
        pathlib.Path.mkdir(MAT_DIR)

    if pathlib.Path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            version_str = f.readlines()[0]
            try:
                found_version = int(version_str)
            except ValueError:
                found_version = 0
    else:
        found_version = 0

    process = True if found_version < MAT_VERSION else False

    def output_file(input_file: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        *_, cycle, filename = input_file.parts
        output_folder = MAT_DIR.joinpath(cycle)
        output_filename = output_folder.joinpath(filename).with_suffix('.mat')
        return output_folder, output_filename

    if process:
        input_files = [RAW_DATA_DIR.joinpath(f) for f in glob.glob('**/*.trc', root_dir=RAW_DATA_DIR)]

        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath('.'), nargout=0)

        for file in input_files:
            output_folder, output_filename = output_file(file)
            if not pathlib.Path.exists(output_folder):
                pathlib.Path.mkdir(output_folder)

            eng.ReadLeCroyBinaryWaveform(str(file), str(output_filename), nargout=0)

        with open(VERSION_FILE, 'w+') as f:
            f.write(str(MAT_VERSION))


if __name__ == '__main__':
    main()
