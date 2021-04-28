from tensorflow.python.keras.callbacks import EarlyStopping

from src.models import MTLearner, MLP, MTMaster


class Learner(MTLearner):
    def __init__(self):
        def model():
            m = MLP(output_act=None, h_units=[16] * 4)
            m.compile(optimizer='adam', loss='mse')
            return m

        # similar to the default behaviour of the scikit MLP (tol = 1e-4, n_iter_no_change = 10, max_iter = 200)
        es = EarlyStopping(monitor='loss', patience=10, min_delta=1e-4)
        super(Learner, self).__init__(model, epochs=200, callbacks=[es], verbose=False)


class Master(MTMaster):
    pass
