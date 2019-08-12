import numpy as np
import os
import time
from multiprocessing.dummy import Pool as ThreadPool

# Set your opensmile Extracter and path here
exe_opensmile = '/home/runnan/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
path_config   = '/home/runnan/opensmile-2.3.0/config/'

# Set your data path and output path here
# One example for IEMOCAP
data_path = "/mnt/runnan/ER-system/IEMOCAP_full_release/"
folder_output = '/mnt/runnan/ER-system/audio_features_ComParE2016/'  # output folder
conf_smileconf = path_config + './ComParE_2016.conf'
sessions  = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

# Extractor set-ups
opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'
outputoption = '-lldcsvoutput'

if not os.path.exists(folder_output):
    os.mkdir(folder_output)

framerate = 16000

def feature_extract(fn):
    print(fn)
    infilename = path_to_wav + fn
    instname = os.path.splitext(fn)[0]
    outfilename = save_path + '/' + instname + '.csv'
    opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + infilename + ' ' + outputoption \
                     + ' ' + outfilename + ' -instname ' + instname + ' -output ?'
    os.system(opensmile_call)
    time.sleep(0.01)

for session in sessions:
    path_to_wav = data_path + session + '/dialog/wav/'
    files = os.listdir(path_to_wav)
    save_path = folder_output + session
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    pool = ThreadPool()
    pool.map(feature_extract, files)
    pool.close()
    pool.join()

os.remove('smile.log')

