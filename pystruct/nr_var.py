class nr_var(object):
    def __init__(self, mu, sigma):
        """
        Initialize a force object. 
        Inputs: 
        mu: The mean for the variable. 
        sigma: The variable standard deviation. 
        """
        self._sigma = sigma
        self._mu = mu
    
    @property
    def sigma(self):
        return deepcopy(self._sigma)
    @property
    def mu(self):
        return deepcopy(self._mu)
    @property
    def list(self):
        return [self.mu, self.sigma]

