import tqdm
import tensorflow as tf

MAX_UPDATE_STEPS = 10


def run_training(dis, gen, n_steps=1000, n_logging=100, session=None,
                 dis_steps=None, ind=None):
    ''' Run GAN training

    Args:
        dis: discriminator trainer.
        gen: generator trainer.
        n_steps: maximum number of training steps.
        n_logging: how often to update the progress bar with the metrics.
        dis_steps: number of discriminator steps to perform.
        ind: indicator displayed in the progress bar.
    '''
    def print_losses(progress_bar, session):
        dis_step, dis_loss, gen_step, gen_loss = \
            session.run([dis.train_step, dis.loss, gen.train_step, gen.loss])
        info = '' if ind is None else '{} | '.format(ind)
        info = info + 'step(d/g) {}/{} | dis {:.4f} | gen {:.4f} '.format(
            dis_step, gen_step, dis_loss, gen_loss)
        progress_bar.set_description(info)

    def update_log(steps, progres_bar, session):
        progress_bar.update(1)
        done = steps > 2*n_steps
        if steps % n_logging == 0:
            print_losses(progress_bar, session)
        return done, steps+1

    dynamic_dis = dis_steps is None   # if dis_steps is none, use dynamic

    session = (session if session is not None
               else tf.compat.v1.get_default_session()
               if tf.compat.v1.get_default_session() is not None
               else tf.InteractiveSession())
    dis_loss, gen_loss = session.run([dis.loss, gen.loss])
    progress_bar = tqdm.tqdm(range(2*n_steps))
    done = False
    steps = 0
    # an each update of either network is considered an step
    while not done:
        # optimize discriminator
        if dynamic_dis:
            dis_steps = max(1, int(dis_loss//gen_loss))
            dis_steps = min(dis_steps, MAX_UPDATE_STEPS)
        # dis_steps = (dis_steps if dis.train_step.eval() > 500
        #              else 1)  # only one during first iterations
        for j in range(dis_steps):
            _, dis_loss, gen_loss = session.run([dis.step, dis.loss, gen.loss])
            done, steps = update_log(steps, progress_bar, session)
            if (dis_loss < gen_loss) or done:
                break
        # optimize generator
        gen_steps = max(1, int(gen_loss//dis_loss))
        gen_steps = min(gen_steps, MAX_UPDATE_STEPS)
        for j in range(gen_steps):
            _, dis_loss, gen_loss = session.run([gen.step, dis.loss, gen.loss])
            done, steps = update_log(steps, progress_bar, session)
            if (gen_loss < dis_loss) or done:
                break
        # break if training not stable anymore
        if dis_loss < 1e-2 or gen_loss < 1e-2:
            print('------------------- TRAINING FAILED ----------------------')
            progress_bar.close()
            return False
    progress_bar.close()
    return True
