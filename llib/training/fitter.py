import numpy as np

class Stopper():
    def __init__(
        self, 
        num_prev_steps=10,
        slope_tol=-0.001,
    ):
        """ 
        Class for stopping optimization process based on slope of linear regression
        Parameters
        ----------
        num_prev_steps: int
            Number of points to use for linear regression
        slope_tol: float
            Tolerance for slope of linear regression
        """

        super(Stopper, self).__init__()

        self.slope_tol = slope_tol
        self.num_prev_steps = num_prev_steps

        self.losses = []

    def _append(self, x):
        self.losses.append(x)

    def linear_regression(self, x, y):
        n = np.size(x) # number of elements
        x_mean = np.mean(x)
        y_mean = np.mean(y)
    
        # cross-deviation and deviation about x
        nx = n * x_mean
        xy = np.sum(y*x) - nx * y_mean
        xx = np.sum(x*x) - nx * x_mean
    
        # coefficients
        slope = xy / xx
        const = y_mean - slope*x_mean
    
        return const, slope

    def _get_slope(self):
        
        assert len(self.losses) > 0, 'No losses have been appended yet'

        # get slope of last num_prev_steps
        y = self.losses[-self.num_prev_steps:]
        x = np.arange(len(y))        
        _, slope = self.linear_regression(x, y)
        
        return slope

    def reset(self):
        self.losses = []

    def check(self, x=None):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
            Returns
            -------
                loss: float
                The final loss value
        '''

        stop = False

        if x is not None:
            self._append(x)

        if len(self.losses) > self.num_prev_steps:

            # fit linear regression to last num_
            slope = self._get_slope()

            if abs(slope) < abs(self.slope_tol) and slope >= self.slope_tol:
                stop = True

        return stop
