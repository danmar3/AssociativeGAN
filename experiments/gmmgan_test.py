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

    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    # session = tf.compat.v1.Session()
    graph = tf.Graph()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        operation_timeout_in_ms=1000)

    session = tf.compat.v1.Session(graph=graph, config=config)
    with graph.as_default(), session.as_default():
        if FLAGS.session is not None:
            experiment = ExperimentGMM.restore_session(
                session_path=FLAGS.session,
                dataset_name='celeb_a')
        else:
            experiment = ExperimentGMM()
            if FLAGS.restore is not None:
                experiment.restore(FLAGS.restore)
        for step in tqdm.tqdm(range(int(FLAGS.n_steps))):
            while not experiment.run(n_steps=int(FLAGS.n_steps_save)):
                print('------------------- RESTORING ---------------------')
                experiment.restore()
            experiment.save()
            experiment.visualize()


if __name__ == "__main__":
    main()
