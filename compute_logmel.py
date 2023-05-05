import os
import librosa
import numpy as np
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import argparse
from loguru import logger
from config_global import n_fft, hop_length, n_mels, fmin, fmax, sample_rate, num_cores, remove_codec_from_filename, verbose
import warnings

number_of_files_success = 0
logger.add(f'compute_logmel_sr={sample_rate}.log')

@logger.catch
def remove_codec_substr(filename: str, remove_codec_from_filename: bool = True):
    """Utility function to remove codec substring from audio files in audioset dataset.

    Args:
        filename (str): Full filepath of audio file
        remove_codec_from_filename (bool, optional): If true will remove the codec substring. Defaults to remove_codec_from_filename.

    Returns:
        str: Final filepath to be used.
    """
    output_filename = os.path.basename(filename)
    if remove_codec_from_filename:
        output_filename = output_filename[:output_filename.rindex('_')]+'.wav'
    return output_filename


@logger.catch
def compute_melspec(filename, outdir, audio_segment_length):
    global number_of_files_success
    save_path = os.path.join(outdir, remove_codec_substr(filename,
                remove_codec_from_filename) + '.npy')
    
    #Prevent generating melspec when .npy already exists
    if not args.overwrite_logmelspec and os.path.exists(save_path):
        return
    try:
        wav = librosa.load(filename, sr=sample_rate)[0]
    except:
        return
    
    if(audio_segment_length != -1 and audio_segment_length != 0):
        wav = wav[:sample_rate*audio_segment_length]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            melspec = librosa.feature.melspectrogram(
                wav,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax)
        logmel = librosa.core.power_to_db(melspec)
        np.save(save_path, logmel)
        if verbose:
            logger.success(save_path)
        number_of_files_success+=1
    except ValueError:
        print('ERROR IN:', filename)
        logger.error(f"{filename} - {save_path}")


@logger.catch
def main(input_path, output_path, audio_segment_length):
    logger.info(f"PARAMS:")
    logger.info(f"n_fft = {n_fft}")
    logger.info(f"hop_length = {hop_length}")
    logger.info(f"n_mels = {n_mels}")
    logger.info(f"fmin = {fmin}")
    logger.info(f"fmax = {fmax}")
    logger.info(f"sample_rate = {sample_rate}")
    logger.info(f"num_cores = {num_cores}")
    logger.info(f"remove_codec_from_filename = {remove_codec_from_filename}")
    logger.info(f"overwrite_logmelspec = {args.overwrite_logmelspec}")
    logger.info(f'Starting computing logmels using above params.')
    file_list = glob(input_path + '/*.wav')
    os.makedirs(output_path, exist_ok=True)
    
    try:
        _ = Parallel(n_jobs=num_cores)(
            delayed(lambda x: compute_melspec(
                x, output_path, audio_segment_length))(x)
            for x in tqdm(file_list))
    except:
        # Add a fallback to using threading backend if the primary backend fails
        print("Using threading backend")
        _ = Parallel(n_jobs=num_cores,backend='threading')(
            delayed(lambda x: compute_melspec(
                x, output_path, audio_segment_length))(x)
            for x in tqdm(file_list))
    
    global number_of_files_success
    logger.success(f'Finished computing logmels using sr = {sample_rate}, total successfully converted to logmels = {number_of_files_success}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input and Output Paths')
    parser.add_argument('input_path', type=str,
                        help="Specifies directory of audio files.")
    parser.add_argument('output_path', type=str,
                        help="Specifies directory for generated spectrograms.")
    parser.add_argument('-a', '--audio_segment_length', type=int,
                        help="Specifies length of audio segment to extract from each audio file. Default -1(Consider full length audio).", default=-1)
    parser.add_argument('-o', '--overwrite_logmelspec', type=bool, help="Specifies whether logmelspec will be generated if it already exists (setting this to False will prevent generation when a file already exists)", default=False)
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.audio_segment_length)
