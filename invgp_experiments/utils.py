import os
import re
from glob import glob
import numpy as np


def get_next_filename(path, base_filename="data"):
    if not os.path.exists(path):
        os.makedirs(path)
    largest_existing_number = max([int(re.findall(r"\d+", fn)[-1]) for fn in glob(f"{path}/{base_filename}*")] + [0])
    return f"{path}/{base_filename}{largest_existing_number + 1}.json"


def plot_1d_model(ax, m, *, data=None):
    D = m.inducing_variable.Z.numpy().shape[1]
    if data is not None:
        X, Y = data[0], data[1]
        ax.plot(X, Y, "x")

    data_inducingpts = np.vstack((X if data else np.zeros((0, D)), m.inducing_variable.Z.numpy()))
    pX = np.linspace(np.min(data_inducingpts) - 1.0, np.max(data_inducingpts) + 1.0, 300)[:, None]
    pY, pYv = m.predict_f(pX)

    (line,) = ax.plot(pX, pY, lw=1.5)
    col = line.get_color()
    ax.plot(pX, pY + 2 * pYv ** 0.5, col, lw=1.5)
    ax.plot(pX, pY - 2 * pYv ** 0.5, col, lw=1.5)
    ax.plot(m.inducing_variable.Z.numpy(), np.zeros(m.inducing_variable.Z.numpy().shape), "k|", mew=2)


def initialize_with_trained_params(new_model, trained_model):
    try:  # non-invariant case
        new_model.kernel.lengthscales.assign(trained_model.kernel.lengthscales.numpy())
        new_model.kernel.variance.assign(trained_model.kernel.variance.numpy())
        new_model.likelihood.variance.assign(trained_model.likelihood.variance.numpy())
        new_model.inducing_variable.Z.assign(trained_model.inducing_variable.Z)
        new_model.q_mu.assign(trained_model.q_mu)
        new_model.q_sqrt.assign(trained_model.q_sqrt)
    except:  # invariant case
        new_model.kernel.basekern.lengthscales.assign(trained_model.kernel.basekern.lengthscales.numpy())
        new_model.kernel.basekern.variance.assign(trained_model.kernel.basekern.variance.numpy())
        new_model.likelihood.variance.assign(trained_model.likelihood.variance.numpy())
        new_model.inducing_variable.Z.assign(trained_model.inducing_variable.Z)
        new_model.q_mu.assign(trained_model.q_mu)
        new_model.q_sqrt.assign(trained_model.q_sqrt)
