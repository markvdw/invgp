from invgp_experiments.mnist import FullBatchMnist

exp = FullBatchMnist(M=1000, dataset_size=60000, inducing_variable_trainable=True, optimisation_method="coord-ascent")
try:
    exp.load()
except FileNotFoundError:
    exp.run()
    exp.save()

print(exp.test_metrics())
