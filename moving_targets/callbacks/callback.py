class Callback:
    def __init__(self):
        super(Callback, self).__init__()

    def on_process_start(self, macs, x, y, val_data):
        pass

    def on_process_end(self, macs, x, y, val_data):
        pass

    def on_pretraining_start(self, macs, x, y, val_data):
        self.on_iteration_start(macs, x, y, val_data, 0)
        self.on_training_start(macs, x, y, val_data, 0)

    def on_pretraining_end(self, macs, x, y, val_data):
        self.on_training_end(macs, x, y, val_data, 0)
        self.on_iteration_end(macs, x, y, val_data, 0)

    def on_iteration_start(self, macs, x, y, val_data, iteration):
        pass

    def on_iteration_end(self, macs, x, y, val_data, iteration):
        pass

    def on_training_start(self, macs, x, y, val_data, iteration):
        pass

    def on_training_end(self, macs, x, y, val_data, iteration):
        pass

    def on_adjustment_start(self, macs, x, y, val_data, iteration):
        pass

    def on_adjustment_end(self, macs, x, y, adjusted_y, val_data, iteration, **kwargs):
        pass
