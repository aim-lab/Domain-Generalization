class DynamicWeight:
    def __init__(self, mse_loss_initial, aug_loss_initial, supp_loss_initial):
        self.supp_loss_initial = supp_loss_initial
        self.aug_loss_initial = aug_loss_initial
        self.mse_loss_initial = mse_loss_initial

        self.curr_epoch = 0

    def update_loss_weight(self):
        self.curr_epoch += 1
        self.mse_loss_initial *= 0.95

        if self.curr_epoch < 20:
            self.supp_loss_initial *= 1.15
            self.aug_loss_initial *= 1.15
        else:
            self.supp_loss_initial *= 0.9
            self.aug_loss_initial *= 0.9

        return {
            'mse_loss': self.mse_loss_initial,
            'reg_loss': 1.0,
            'supp_loss': self.supp_loss_initial,
            'aug_loss': self.aug_loss_initial,
        }
