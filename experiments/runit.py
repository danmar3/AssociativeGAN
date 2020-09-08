import os
import time
import tqdm
import GPUtil
import argparse
import acgan
import json
import tensorflow as tf
from acgan.main import (
    ExperimentGMM, ExperimentWACGAN, ExperimentWACGAN_V2, ExperimentWACGAN_Dev,
    ExperimentBiGmmGan, ExperimentExGan)

FLAGS = None
EXPERIMENTS = {
    'gmmgan': ExperimentGMM,
    'wacgan': ExperimentWACGAN,
    'wacganV2': ExperimentWACGAN_V2,
    'wacganDev': ExperimentWACGAN_Dev,
    'bigmmgan': ExperimentBiGmmGan,
    'exgan': ExperimentExGan}

# python3 gmmgan_test.py --n_steps=100 --n_steps_save=5 --gpu=6 --session="tmp/gmmgan/session_20190925:0443"


def main():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', default=None,
        help=("checkpoint to load values of previous weights."))
    parser.add_argument('--session', default=None,
                        help=("previous session path to load."))
    parser.add_argument('--n_steps', default=100,
                        help=("number of steps to run"))
    parser.add_argument('--n_steps_save', default=10,
                        help=("number of steps to run before save"))
    parser.add_argument('--gpu', default=None,
                        help='gpu to use')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='use cpu only')
    parser.add_argument('--dataset', default='celeb_a',
                        help='dataset to use')
    parser.add_argument('--model', default='gmmgan',
                        help='model to test')
    parser.add_argument('--datadir', default=None,
                        help='datasets path')
    parser.add_argument('--clean', default=False, action='store_true',
                        help='clearn directory before start')
    parser.add_argument(
        '--indicator', default=None,
        help='indicator printed during logging to identify the experiment')

    FLAGS = parser.parse_args()
    if FLAGS.cpu:
        print("USING CPU ONLY !!!!")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif FLAGS.gpu is None:
        gpu_ids = GPUtil.getAvailable(maxLoad=0.1, maxMemory=0.1, limit=1)
        while not gpu_ids:
            gpu_ids = GPUtil.getAvailable(maxLoad=0.1, maxMemory=0.1, limit=1)
            print('GPUs are busy, waiting...')
            time.sleep(30)
        print('\n\n---> Using GPU: {} \n\n'.format(gpu_ids[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    else:
        gpu_ids = GPUtil.getAvailable(maxLoad=0.1, maxMemory=0.1, limit=10)
        while int(FLAGS.gpu) not in gpu_ids:
            gpu_ids = GPUtil.getAvailable(maxLoad=0.1, maxMemory=0.1, limit=10)
            print('GPU is busy, waiting...')
            time.sleep(30)
        print('\n\n---> Using GPU: {} \n\n'.format(FLAGS.gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    acgan.data.set_datadir(FLAGS.datadir)
    # cleanup
    assert FLAGS.model in EXPERIMENTS, \
        'Invalid model. Supported models are: {}'\
        ''.format(EXPERIMENTS.keys())
    if FLAGS.clean:
        acgan.saver.clean_folder(data_dir=os.path.join('tmp', FLAGS.model))
    # create graph
    graph = tf.Graph()
    # config = tf.ConfigProto(
    #    # allow_soft_placement=True,
    #    # log_device_placement=True,
    #    # operation_timeout_in_ms=1000
    #    )

    session = tf.compat.v1.Session(graph=graph)
    with graph.as_default(), session.as_default():
        # instantiate experiment
        if FLAGS.session is not None:
            experiment = EXPERIMENTS[FLAGS.model].restore_session(
                session_path=FLAGS.session,
                dataset_name=FLAGS.dataset,
                indicator=FLAGS.indicator)
        else:
            experiment = EXPERIMENTS[FLAGS.model](
                dataset_name=FLAGS.dataset,
                indicator=FLAGS.indicator)
            if FLAGS.checkpoint is not None:
                experiment.params['run']['n_start'] = 0
                experiment.restore(FLAGS.checkpoint)
        # run training
        for step in tqdm.tqdm(range(int(FLAGS.n_steps))):
            while not experiment.run(n_steps=int(FLAGS.n_steps_save)):
                print('------------------- RESTORING ---------------------')
                experiment.restore()
            experiment.save()
            experiment.visualize(save=True)


if __name__ == "__main__":
    main()
