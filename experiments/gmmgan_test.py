import os
import tqdm
import argparse
import acgan
from acgan.main import ExperimentGMM

FLAGS = None


def main():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', default=None,
                        help=("checkpoint to load values of previous weights"))
    parser.add_argument('--n_steps', default=100,
                        help=("number of steps to run"))
    parser.add_argument('--n_steps_save', default=10,
                        help=("number of steps to run before save"))
    parser.add_argument('--gpu', default=0, help='gpu to use')

    FLAGS = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

    experiment = ExperimentGMM()
    if FLAGS.restore is not None:
        experiment.restore(FLAGS.restore)
    for step in tqdm.tqdm(range(int(FLAGS.n_steps))):
        experiment.run(n_steps=int(FLAGS.n_steps_save))
        experiment.save()
        experiment.visualize()


if __name__ == "__main__":
    main()
