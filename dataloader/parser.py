import numpy as np
import pickle
import os
import argparse
from .file_utils import load_object, Serialized_Dict
from shutil import copyfile

def find_nearest_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    try:
        if abs(value - array[idx - 1]) < abs(value - array[idx]):
            return idx - 1
        else:
            return idx
    except IndexError:
        return idx - 1
    
def deserialize(pts):
    data = []
    for p in pts:
        data.append({
            'norm_pos': p['norm_pos'],
            'timestamp': p['timestamp'],
            'confidence': p['confidence'],
            'topic': p['topic']
        })
    return data

def check_files(dir):
    if not os.path.exists(os.path.join(dir, 'world.mp4')):
        print('[FILE ERROR] Required: {} not found'.format(os.path.exists(dir, 'world.mp4')))
        return False
    if not os.path.exists(os.path.join(dir, 'pupil_data')):
        print('[FILE ERROR] Required: {} not found'.format(os.path.join(dir, 'pupil_data')))
        return False
    if not os.path.exists(os.path.join(dir, 'world_timestamps.npy')):
        print('[FILE ERROR] Required: {} not found'.format(os.path.join(dir, 'world_timestamps.npy')))
        return False
    print('[FILE INFO] Find all required files from {}'.format(dir))
    return True

def parse(dir):
    pupil_data_path = os.path.join(dir, 'pupil_data')
    print('[PARSER INFO] loading pupil data gaze positions {}'.format(dir))
    pupil_data = [Serialized_Dict(item) for item in load_object(pupil_data_path)["gaze_positions"]]
    pupil_timestamps = [item['timestamp'] for item in pupil_data]
    print('[PARSER INFO] loading world timestamps {}'.format(dir))
    world_timestamps = np.load(os.path.join(dir, 'world_timestamps.npy'))
    print('[PARSER INFO] matching world timestamps with existing pupil data {}'.format(dir))
    index_mapper = [find_nearest_idx(pupil_timestamps, i) for i in world_timestamps]
    # need to write test cases: 1. test sorted? 2. test video frame number? 3. test matching quality
    world_pupil_data = [pupil_data[i] for i in index_mapper]
    world_pupil_data = deserialize(world_pupil_data)
    np.save(os.path.join(dir, 'world_pupil_data.npy'), world_pupil_data)
    print('[PARSER INFO] saved parse pupil information {}'.format(dir))

# modularity testing
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process for Pupil Data')
    parser.add_argument('--data_folder', default='data', type=str, help='pupil data directory for processing gaze information')
    parser.add_argument('--save_folder', default='result', type=str, help='destination directory for post-processed files')
    args = parser.parse_args()

    if check_files(args.data_folder):
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
            copyfile(os.path.join(args.data_folder, 'world.mp4'), os.path.join(args.save_folder, 'world.mp4'))
        pupil_data_path = os.path.join(args.data_folder, 'pupil_data')

        print('loading pupil data gaze positions...')
        pupil_data = [Serialized_Dict(item) for item in load_object(pupil_data_path)["gaze_positions"]]
        pupil_timestamps = [item['timestamp'] for item in pupil_data]
        print('loading world timestamps...')
        world_timestamps = np.load(os.path.join(args.data_folder, 'world_timestamps.npy'))
        print('matching world timestamps with existing pupil data')
        index_mapper = [find_nearest_idx(pupil_timestamps, i) for i in world_timestamps]
        # need to write test cases: 1. test sorted? 2. test video frame number? 3. test matching quality
        world_pupil_data = [pupil_data[i] for i in index_mapper]
        world_pupil_data = deserialize(world_pupil_data)
        np.save(os.path.join(args.save_folder, 'world_pupil_data.npy'), world_pupil_data)
