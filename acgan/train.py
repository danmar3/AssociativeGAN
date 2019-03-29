import tqdm
import tensorflow as tf


def run_training(dis, gen, n_steps=1000, n_logging=100, session=None):
    def print_losses():
        print('step {} | dis {} | gen {}'.format(
            dis.train_step.eval(), dis.loss.eval(), gen.loss.eval()))
    session = (session if session is not None
               else tf.get_default_session()
               if tf.get_default_session() is not None
               else tf.InteractiveSession())

    for i in tqdm.tqdm(range(n_steps)):
        # optimize discriminator
        _, dis_loss, gen_loss = session.run(
            [dis.step, dis.loss, gen.loss])
        # optimize generator
        gen_steps = max(1, int(gen_loss//dis_loss))
        for j in range(gen_steps):
            _, dis_loss, gen_loss = session.run(
                    [gen.step, dis.loss, gen.loss])
        # log training info
        if i % n_logging == 0:
            print_losses()
