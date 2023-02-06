import copy


class ExponentialMovingAverage():

    def __init__(self, model, momentum=0.99, warmup=50):

        super().__init__()

        self.step = 0
        self.warmup = warmup
        self.momentum = momentum

        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    def ema_step(self, step, model):

        if step < self.warmup:
            self.ema_model.load_state_dict(model.state_dict())

        else:

            for ema_params, model_params in zip(
                    self.ema_model.named_parameters(),
                    model.named_parameters()):

                ema_weight, model_weight = ema_params.data, model_params.data

                ema_params.data = (ema_weight * self.momentum +
                                   model_weight * (1 - self.momentum))
