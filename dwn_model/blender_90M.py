VERSION = 'v1.0'
import os
from parlai.core.build_data import download_models, built

def print_blender():
    curr_dir = os.path.dirname(__file__)
    txt_file = os.path.join(curr_dir, 'blender_ascii.txt')
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            print(line[:-1])


def build(datapath, fname, model_type, version):
    opt = {'datapath': datapath}
    opt['model_type'] = model_type
    dpath = os.path.join(datapath, 'models', 'blender', model_type)
    if not built(dpath, version):
        print_blender()
    download_models(opt, [fname], 'blender', version=version, use_model_type=False)


def download(datapath):
    build(datapath, 'BST0B.tgz', model_type='blender_90M', version=VERSION)


download('/usr/local/lib/python3.10/dist-packages/data')
