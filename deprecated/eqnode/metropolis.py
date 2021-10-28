import numpy as np


class MetropolisGauss(object):

    def __init__(self, model, x0, temperature=1.0, noise=0.1,
                 burnin=0, stride=1, nwalkers=1, mapper=None):
        """ Metropolis Monte-Carlo Simulation with Gaussian Proposal Steps

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x)
        x0 : [array]
            Initial configuration
        noise : float
            Noise intensity, standard deviation of Gaussian proposal step
        temperatures : float or array
            Temperature. By default (1.0) the energy is interpreted in reduced units.
            When given an array, its length must correspond to nwalkers, then the walkers
            are simulated at different temperatures.
        burnin : int
            Number of burn-in steps that will not be saved
        stride : int
            Every so many steps will be saved
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.model = model
        self.noise = noise
        self.temperature = temperature
        self.burnin = burnin
        self.stride = stride
        self.nwalkers = nwalkers
        if mapper is None:
            class DummyMapper(object):
                def map(self, X):
                    return X
            mapper = DummyMapper()
        self.mapper = mapper
        self.reset(x0)

    def _proposal_step(self):
        # proposal step
        self.x_prop = self.x + self.noise*np.random.randn(self.x.shape[0], self.x.shape[1])
        self.x_prop = self.mapper.map(self.x_prop)
        self.E_prop = self.model.energy(self.x_prop)

    def _acceptance_step(self):
        # acceptance step
        acc = -np.log(np.random.rand()) > (self.E_prop - self.E) / self.temperature
        self.x = np.where(acc[:, None], self.x_prop, self.x)
        self.E = np.where(acc, self.E_prop, self.E)
        return acc.mean()

    def reset(self, x0):
        # counters
        self.step = 0
        self.traj_ = []
        self.etraj_ = []

        # initial configuration
        self.x = np.tile(x0, (self.nwalkers, 1))
        self.x = self.mapper.map(self.x)
        self.E = self.model.energy(self.x)

        # save first frame if no burnin
        if self.burnin == 0:
            self.traj_.append(self.x)
            self.etraj_.append(self.E)

    @property
    def trajs(self):
        """ Returns a list of trajectories, one trajectory for each walker """
        T = np.array(self.traj_).astype(np.float32)
        return [T[:, i, :] for i in range(T.shape[1])]

    @property
    def traj(self):
        return self.trajs[0]

    @property
    def etrajs(self):
        """ Returns a list of energy trajectories, one trajectory for each walker """
        E = np.array(self.etraj_)
        return [E[:, i] for i in range(E.shape[1])]

    @property
    def etraj(self):
        return self.etrajs[0]

    def run(self, nsteps=1, verbose=0):
        acceptance_rates = []
        for i in range(nsteps):
            self._proposal_step()
            acceptance_rate = self._acceptance_step()
            acceptance_rates.append(acceptance_rate)
            self.step += 1
            if verbose > 0 and i % verbose == 0:
                print('Step', i, '/', nsteps)
            if self.step > self.burnin and self.step % self.stride == 0:
                self.traj_.append(self.x)
                self.etraj_.append(self.E)
        return np.mean(acceptance_rates)