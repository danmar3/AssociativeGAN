import tqdm
import tensorflow as tf

MAX_UPDATE_STEPS = 100


def run_training(dis, gen, n_steps=1000, n_logging=100, session=None):
    def print_losses(progress_bar):
        info = 'step {} | dis {:.4f} | gen {:.4f} '.format(
            dis.train_step.eval(), dis.loss.eval(), gen.loss.eval())
        progress_bar.set_description(info)

    def update_log(steps, progres_bar):
        progress_bar.update(1)
        done = steps > 2*n_steps
        if steps % n_logging == 0:
            print_losses(progress_bar)
        return done, steps+1

    session = (session if session is not None
               else tf.get_default_session()
               if tf.get_default_session() is not None
               else tf.InteractiveSession())
    dis_loss, gen_loss = session.run([dis.loss, gen.loss])
    progress_bar = tqdm.tqdm(range(2*n_steps))
    done = False
    steps = 0
    # an each update of either network is considered an step
    while not done:
        # optimize discriminator
        dis_steps = max(1, int(dis_loss//gen_loss))
        dis_steps = min(dis_steps, MAX_UPDATE_STEPS)
        dis_steps = (dis_steps if dis.train_step.eval() > 500
                     else 1)  # only one during first iterations
        for j in range(dis_steps):
            _, dis_loss, gen_loss = session.run([dis.step, dis.loss, gen.loss])
            done, steps = update_log(steps, progress_bar)
            if (dis_loss < gen_loss) or done:
                break
        # optimize generator
        gen_steps = max(1, int(gen_loss//dis_loss))
        gen_steps = min(gen_steps, MAX_UPDATE_STEPS)
        for j in range(gen_steps):
            _, dis_loss, gen_loss = session.run([gen.step, dis.loss, gen.loss])
            done, steps = update_log(steps, progress_bar)
            if (gen_loss < dis_loss) or done:
                break
    progress_bar.close()
