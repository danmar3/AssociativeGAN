import os
import tqdm
import argparse
import acgan
import json
import tensorflow as tf
from acgan.main import ExperimentGMM

FLAGS = None

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
    parser.add_argument('--gpu', default=0, help='gpu to use')
    parser.add_argument('--dataset', default='celeb_a', help='dataset to use')
    parser.add_argument('--model', default='gmmgan', help='model to test')
    parser.add_argument('--datadir', default=None, help='datasets path')
    parser.add_argument('--clean', default=False, action='store_true',
                        help='clearn directory before start')
    parser.add_argument(
        '--indicator', default=None,
        help='indicator printed during logging to identify the experiment')

    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    acgan.data.set_datadir(FLAGS.datadir)
    # cleanup
    assert FLAGS.model == 'gmmgan', \
        'only gmmgan model is supported at the moment'
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
            experiment = ExperimentGMM.restore_session(
                session_path=FLAGS.session,
                dataset_name=FLAGS.dataset,
                indicator=FLAGS.indicator)
        else:
            experiment = ExperimentGMM(dataset_name=FLAGS.dataset,
                                       indicator=FLAGS.indicator)
            if FLAGS.checkpoint is not None:
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
