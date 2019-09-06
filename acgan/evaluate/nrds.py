import numpy as np


class NRDS:

    @staticmethod
    def score(*loss_curves, **named_loss_curves):
        """
        Compute area undet the curv
        :param disc_loss_curve: pass 1d numpy arrays as unamed args
        :return:
        """
        if len(loss_curves) > 0 and len(named_loss_curves) > 0:
            raise ValueError("Use of positional arguments and keyword arguments are mutually exclusive")

        if len(loss_curves) > 0:
            l_areas = np.array([np.trapz(l_arr) for l_arr in loss_curves])
            nrds = l_areas / l_areas.sum()
        else:
            l_area_map = {l_name: np.trapz(l_arr) for l_name, l_arr in named_loss_curves.items()}
            total_l = np.sum(list(l_area_map.values()))
            nrds = {ln: l/total_l for ln, l in l_area_map.items()}

        return nrds


score = NRDS.score
