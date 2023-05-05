import numpy as np

__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

use_resampled_data = True

model_archs = ['mobilenetv2', 'pann_cnn10', 'pann_cnn14', "mobilenetv3"]
class_mapping = {}
if use_resampled_data:
    class_mapping['breaking'] = 0
    class_mapping['crowd'] = 1
    class_mapping['crying'] = 2
    class_mapping['explosion'] = 3
    class_mapping['gunshot'] = 4
    class_mapping['motor'] = 5
    class_mapping['siren'] = 6
    class_mapping['speech'] = 7
    #class_mapping['others'] = 8
    class_mapping['silence'] = 8
else:
    class_mapping['breaking'] = 0
    class_mapping['chatter'] = 1
    class_mapping['crying_sobbing'] = 2
    class_mapping['emergency_vehicle'] = 3
    class_mapping['explosion'] = 4
    class_mapping['gunshot_gunfire'] = 5
    class_mapping['motor_vehicle_road'] = 6
    class_mapping['screaming'] = 7
    class_mapping['siren'] = 8
    class_mapping['others'] = 9
    
num_workers = 2
feature_type = 'logmelspec'
num_bins = 128
resize = True
learning_rate = 0.001
amsgrad = True
verbose = True
patience = 5
epochs = 50
early_stopping = 10
gpu = False
channels = 2
length_full_recording = 10
audio_segment_length = 3
seed = 42

#                           441,000        160,000
# nfft/window_len           2560        7056
# hop_len                   694         1912
# num_frames                656         84
sample_rate = 44100
threshold = 0.9
# n_fft = (2560*sample_rate)//44100
# n_fft = 2048
# hop_length = 512
n_fft = (2560*sample_rate)//44100
hop_length = (694*sample_rate)//44100
n_mels = 128
fmin = 20
fmax = 8000
# num_frames = 200
num_frames = int(np.ceil(sample_rate*length_full_recording/hop_length))

permutation = [0, 1, 2, 3, 4]
workspace = '/notebooks/sound-event-classification/audioset'
# target_names = ['breaking', 'chatter', 'crying_sobbing', 'emergency_vehicle',
#                 'explosion', 'gunshot_gunfire', 'motor_vehicle_road', 'screaming', 'siren', 'others']
target_names = list(class_mapping.keys())
num_classes = len(target_names)
# for balancedbatchsampler, for every batch to have equal number of samples, the size of each batch should be a multiple of the num of classes
#batch_size = 64
#batch_size = 128 #this works for paperspace Free-A5000
batch_size = 288 #this works for paperspace Free-A6000
#batch_size = 512 #Free-A100-80G
grad_acc_steps = 1

# voting = 'simple_average'
voting = 'weighted_average'
weights = [2, 3, 5]
sum_weights = sum(weights)
normalised_weights = np.array(weights)/sum_weights

# CBAM
use_cbam = True
cbam_channels = 512
cbam_reduction_factor = 16
cbam_kernel_size = 7

# PNA
use_pna = False

# MEDIAN FILTERING
use_median_filter = False

# paperspace
pann_cnn10_encoder_ckpt_path = '/notebooks/sound-event-classification/audioset/model/Cnn10_mAP=0.380.pth'
pann_cnn14_encoder_ckpt_path = '/notebooks/sound-event-classification/audioset/model/Cnn14_mAP=0.431.pth'
model_arch = 'mobilenetv2'
resume_training = 'yes'

# FDY
use_fdy = True
use_tdy = False
n_basis_kernels = 4
temperature = 31
if use_fdy:
    pool_dim = "time"  # FDY use "time", for TDY use "freq"
elif use_tdy:
    pool_dim = "freq"
