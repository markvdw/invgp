from dataclasses import dataclass
from glob import glob
from typing import Optional

import json_tricks
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from invgp_experiments.datasets import load_mnist, load_mnist_rot
from invgp_experiments.utils import get_next_filename

import gpflow
from gpflow.utilities import set_trainable

gpflow.config.set_default_positive_minimum(1e-6)


@dataclass
class Experiment:
    storage_path: str
    base_filename: Optional[str] = "data"

    # Populated during object life
    model = None
    trained_parameters = None

    _X = None
    _Y = None
    _X_test = None
    _Y_test = None

    def load_data(self):
        raise NotImplementedError

    def setup_model(self):
        raise NotImplementedError

    def init_params(self):
        raise NotImplementedError

    def run_optimisation(self):
        raise NotImplementedError

    def run(self):
        self.setup_model()
        self.init_params()
        self.run_optimisation()

    @property
    def X(self):
        if self._X is None:
            self.load_data()
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self.load_data()
        return self._Y

    @property
    def X_test(self):
        if self._X_test is None:
            self.load_data()
        return self._X_test

    @property
    def Y_test(self):
        if self._Y_test is None:
            self.load_data()
        return self._Y_test

    @property
    def store_variables(self):
        return [k for k in list(self.__dict__.keys())
                if k[0] != '_' and
                k not in ["storage_path", "base_filename", "model"]]

    @property
    def load_match_variables(self):
        return [k for k in self.store_variables if k not in ["trained_parameters"]]

    def save(self):
        store_dict = {k: v for k, v in self.__dict__.items() if k in self.store_variables}
        json_tricks.dump(store_dict, get_next_filename(self.storage_path, "data"))

    def load(self, filename=None):
        if filename is None:
            # Find run with similar run parameters
            existing_runs = []
            for fn in glob(f"{self.storage_path}/{self.base_filename}*"):
                existing_runs.append((json_tricks.load(fn), fn))

            matching_runs = [(loaded_dict, fn) for loaded_dict, fn in existing_runs
                             if all([self.__dict__[k] ==
                                     (loaded_dict[k] if k in loaded_dict else self.__dataclass_fields__[k].default)
                                     for k in self.load_match_variables])]
        else:
            matching_runs = [(json_tricks.load(filename), filename)]

        if len(matching_runs) == 1:
            print(f"Loading from `{matching_runs[0][1]}`...")
            for k, v in matching_runs[0][0].items():
                setattr(self, k, v)
            self.setup_model()
            gpflow.utilities.multiple_assign(self.model, self.trained_parameters)
        elif len(matching_runs) == 0:
            raise FileNotFoundError("No matching run found.")
        else:
            raise AssertionError("Only one run of an experiment should be present.")


@dataclass
class FullBatchMnist(Experiment):
    # Run parameters
    dataset_size: Optional[int] = -1
    dataset_name: Optional[str] = "mnist"
    storage_path: Optional[str] = "./results/FullBatchMnist"

    data_minibatch_size: Optional[int] = None
    M: Optional[int] = 500
    kernel_name: Optional[str] = "SquaredExponential"

    optimisation_method: Optional[str] = "joint"  # joint | coord-ascent
    inducing_variable_trainable: Optional[bool] = True

    def setup_model(self):
        if self.kernel_name == "SquaredExponential":
            kernel = gpflow.kernels.SquaredExponential()
        else:
            raise NotImplementedError
        M = min(self.M, self.dataset_size)
        inducing_variable = gpflow.inducing_variables.InducingPoints(np.zeros((M, 28 * 28)))
        self.model = gpflow.models.SGPR((self.X, self.Y), kernel, inducing_variable=inducing_variable)

    def init_params(self):
        Z = self.X[rnd.permutation(len(self.X))[:self.M], :]
        self.model.inducing_variable.Z = gpflow.Parameter(Z)
        set_trainable(self.model.inducing_variable, self.inducing_variable_trainable)

        self.model.kernel.lengthscales.assign(75)

    def run_optimisation(self):
        try:
            if self.optimisation_method == "joint":
                opt = gpflow.optimizers.Scipy()
                opt.minimize(tf.function(lambda: -self.model.elbo()), self.model.trainable_variables,
                             options=dict(maxiter=1000, disp=True))
            elif self.optimisation_method == "coord-ascent":
                for i in range(100):
                    set_trainable(self.model.inducing_variable, False)
                    set_trainable(self.model.kernel, True)
                    set_trainable(self.model.likelihood, True)
                    opt = gpflow.optimizers.Scipy()
                    opt.minimize(tf.function(lambda: -self.model.elbo()),
                                 self.model.trainable_variables, options=dict(maxiter=100, disp=True))

                    set_trainable(self.model.inducing_variable, True)
                    set_trainable(self.model.kernel, False)
                    set_trainable(self.model.likelihood, False)
                    opt = gpflow.optimizers.Scipy()
                    opt.minimize(tf.function(lambda: -self.model.elbo()),
                                 self.model.trainable_variables, options=dict(maxiter=120, disp=True))
            else:
                raise NotImplementedError
        except KeyboardInterrupt:
            print("Optimisation cancelled by user...")
        finally:
            self.trained_parameters = gpflow.utilities.read_values(self.model)

    def load_data(self):
        if self.dataset_name == "mnist":
            (self._X, Y), (self._X_test, Y_test) = load_mnist()
        elif self.dataset_name == "mnist-rot":
            (self._X, Y), (self._X_test, Y_test) = load_mnist_rot()
        else:
            raise NotImplementedError
        self._Y = np.eye(10)[Y[:, 0]].astype(gpflow.config.default_float())
        self._Y_test = np.eye(10)[Y_test[:, 0]].astype(gpflow.config.default_float())

    def test_metrics(self):
        m, v = self.model.predict_f(self.X_test)
        accuracy = np.mean(m.numpy().argmax(1) == self.Y_test.argmax(1))
        return accuracy, None

    @property
    def X(self):
        return super().X[:self.dataset_size, :]

    @property
    def Y(self):
        return super().Y[:self.dataset_size, :]
