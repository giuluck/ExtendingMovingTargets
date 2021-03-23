class Logger:
    def __init__(self):
        super(Logger, self).__init__()

    def on_process_start(self, macs):
        pass

    def on_process_end(self, macs):
        pass

    def on_pretraining_start(self, macs):
        pass

    def on_pretraining_end(self, macs):
        pass

    def on_iteration_start(self, macs, idx):
        pass

    def on_iteration_end(self, macs, idx):
        pass

    def on_training_start(self, macs):
        pass

    def on_training_end(self, macs, x, y, val_x, val_y):
        pass

    def on_adjustment_start(self, macs):
        pass

    def on_adjustment_end(self, macs, x, y, adj_y):
        pass
