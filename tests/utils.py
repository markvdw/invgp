

def initialize_with_trained_params(new_model, trained_model):
	try: # non-invariant case
	    new_model.kernel.lengthscales.assign(trained_model.kernel.lengthscales.numpy())
	    new_model.kernel.variance.assign(trained_model.kernel.variance.numpy())
	    new_model.likelihood.variance.assign(trained_model.likelihood.variance.numpy())
	    new_model.inducing_variable.Z.assign(trained_model.inducing_variable.Z)
	    new_model.q_mu.assign(trained_model.q_mu)
	    new_model.q_sqrt.assign(trained_model.q_sqrt)
	except: # invariant case
	    new_model.kernel.basekern.lengthscales.assign(trained_model.kernel.basekern.lengthscales.numpy())
	    new_model.kernel.basekern.variance.assign(trained_model.kernel.basekern.variance.numpy())
	    new_model.likelihood.variance.assign(trained_model.likelihood.variance.numpy())
	    new_model.inducing_variable.Z.assign(trained_model.inducing_variable.Z)
	    new_model.q_mu.assign(trained_model.q_mu)
	    new_model.q_sqrt.assign(trained_model.q_sqrt)