import os
import sys
from audiomentations import *
import numpy as np
import librosa
from matplotlib import pyplot as plt
from pathlib import Path
import soundfile as sf
from glob import glob
from tqdm import tqdm
import argparse
import re

def augment_default(in_path, output, extension):
    file_list = []
    if not os.path.isdir(in_path):
        file_list.append(in_path)
    else:
        file_list = glob(os.path.join(in_path,f"*.{extension})"))
                         
    # Set output path to be the same directory of first input audio sample
    if not output and len(file_list) > 0:
        path = Path(file_list[0])
        out_path = str(path.parent.absolute())
    else:
        out_path = output
                         
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
                         
    for i in file_list:
        try:
            wav, sr = librosa.load(i, sr=16000, mono=False)
            augment = Compose([
                AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.005, p=0.5),
                AirAbsorption(p=0.8),
                SevenBandParametricEQ(p=0.8),
            ])
            augmented_wav = augment(samples=wav.astype(np.float32), sample_rate=16000)
            try:
                sf.write(f"{out_path}/{os.path.basename(i)[:-4:]}-augmented.wav", data=augmented_wav, samplerate=16000)
            except:
                sf.write(f"{out_path}/{os.path.basename(i)[:-4:]}-augmented.wav", data=augmented_wav.T, samplerate=16000)
        except Exception as e:
            print(str(e))


def slugify(s):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    s = s.lower().strip()
    s = re.sub(r'[^\w\s-]', '', s)
    s = re.sub(r'[\s_-]+', '_', s)
    s = re.sub(r'^-+|-+$', '', s)
    return s
	
def my_fmt(x):
    return '({:.0f})'.format(total*x/100)

def gen_dist_chart(in_path, output, title):
    global total
    print(os.path.join(in_path,'*'))
    file_list = glob(os.path.join(in_path,'*'))

    val_mapping = {}
    for i in tqdm(file_list):
        split = os.path.basename(i).split("-")
        classname = split[0]
        if "other" in os.path.basename(i):
            classname = "others"

        if not classname in val_mapping:
            val_mapping[classname] = 1
        else:
            val_mapping[classname] = val_mapping[classname] + 1


	# remove labels from chart
    #pop_keys = ['others']
    #for i in pop_keys:
    #    val_mapping.pop(i)

    
    fig = plt.figure(figsize =(10, 7))
    
    #if dist_type == "train":
    #    title = "Train set distribution (70%)"
    #elif dist_type == "test":
    #    title = "Test set distribution (15%)"
    #else:
    #    title = "Valid set distribution (15%)"
        
    total = sum(list(val_mapping.values()))
    plt.title(title)
    plt.pie(list(val_mapping.values()), labels = list(val_mapping.keys()), autopct=my_fmt)
	
    if output is None:
        plt.savefig(f"{slugify(title)}.png")
    else:
        plt.savefig(os.path.join(output, f"{slugify(title)}.png")[:-1:])
	

def main():
    print(f"Action: {args.action}")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    if args.action == "augment":
        augment_default(args.input_path, args.output_path, args.extension)
    elif args.action == "chart":
        gen_dist_chart(args.input_path, args.output_path, args.title)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for dataset utility')
    parser.add_argument('input_path', type=str,
                        help="Specifies file path or directory of audio file/files")
    parser.add_argument('-o','--output_path', type=str,
                        help="Specifies output directory", default=None)
    parser.add_argument('-a', '--action',
                        type=str, help="Type of action to perform on the dataset. Options are ('augment','chart')",default="augment")
    parser.add_argument('-t', '--title',
                        type=str, help="Title of generated distribution chart (applicable when -a is set to 'chart'", default="Untitled Distribution Chart")
    parser.add_argument('-ext', '--extension', type=str,
                        help="Specifies file extension of audio samples", default="wav")
    args = parser.parse_args()

    main()