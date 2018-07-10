"""Defines the KineticModel class, used for kinetic models that can be
fit to experimental data or simulated with a given set of parameters.

"""
import itertools
import scipy.integrate, scipy.optimize, numpy as np

class KineticModel:
    """Defines a kinetic model (as a system of differential equations).

    Attributes:
        name (string): Short form name used to identify model.
        description (string): Multiline description of model.
        kin_sys (list of equations): List of differential equations that
            will be solved.
        ks_guesses (list of floats): Initial guesses for k's and K's.
        starting_concs_guesses (list of floats): Initial guesses for
            starting concs to be optimized.
        starting_concs_constant (list of floats): Concentrations that
            will be held constant.
        parameter_names (list of strings): Labels used for the
            parameters.
        legend_names (list of strings): Labels used for the
            concentrations.
        top_plot (list of ints): Concentrations that should be plotted
            on top plot of output.
        bottom_plot (list of ints):
        sort_order (list of ints): Translates order of columns in exp
            input into order of concentrations as defined in model.
        int_eqn (list of functions): Functions of concentration and
            parameters that will be integrated, typically passed
            as lambda functions of lists of concentrations and 
            parameters (e.g., lambda cs, ks: ...)
        int_eqn_desc (list of strings): Descriptions of int_eqn.
        bounds (tuple of floats, default (0, np.inf)): bounds for 
            parameters.

    Properties:
        num_concs: Number of independent concentrations
        num_ks: Number of k's and K's.
        num_params: Number of variable parameters.
        len_params: Length of the longest parameter name.
        len_int_eqn_desc: Length of the longest integral description.

    """

    MAX_STEPS = 10000 # Maximum steps in scipy.integrate.odeint
    
    def __init__(self,
                 name,
                 description,
                 kin_sys,
                 ks_guesses, 
                 starting_concs_guesses, 
                 starting_concs_constant, 
                 parameter_names,
                 legend_names,
                 top_plot,
                 bottom_plot,
                 sort_order,
                 int_eqn = [],
                 int_eqn_desc = [],
                 bounds = (0, np.inf)):
        
        self.name = name
        self.description = description
        self.kin_sys = kin_sys
        self.ks_guesses = ks_guesses
        self.starting_concs_guesses = starting_concs_guesses
        self.starting_concs_constant = starting_concs_constant
        self.parameter_names = parameter_names
        self.legend_names = legend_names
        self.top_plot = top_plot
        self.bottom_plot = bottom_plot
        self.sort_order = sort_order
        self.int_eqn = int_eqn
        self.int_eqn_desc = int_eqn_desc
        self.bounds = bounds

    @property
    def num_concs(self):
        return len(self.starting_concs_guesses) + len(self.starting_concs_constant)    

    @property
    def num_ks(self):
        return len(self.ks_guesses)

    @property
    def num_params(self):
        return self.num_ks + len(self.starting_concs_guesses)

    @property
    def len_params(self):
        return max(len(n) for n in self.parameter_names)

    @property
    def len_int_eqn_desc(self):
        return max(len(n) for n in self.int_eqn_desc)

    def simulate(self, ks, concs, num_points, max_time, integrate=True):
        """Using _solved_kin_sys, solves the system of diff. equations for a
        with a given number of points, maximum time, and with optional
        integration.

        """
        smooth_ts_out, deltaT = np.linspace(0, max_time, num_points, retstep=True)
        smooth_curves_out = self._solved_kin_sys(np.append(concs,
                                                 self.starting_concs_constant), 
                                                 ks, smooth_ts_out)

        if integrate:
            integrals = {}
            for i in range(len(self.int_eqn)):
                integral = 0
                for t in smooth_curves_out:
                    integral += self.int_eqn[i](t, ks) * deltaT
                integrals[self.int_eqn_desc[i]] = integral
        else:
            integrals = None

        return smooth_ts_out, smooth_curves_out, integrals

    def fit_to_model(self, exp_times, exp_concs, total_points=None,
        error_function=(lambda exp, calc: exp - calc), monitor=False):
        """Performs a fit to a set of experimental times and concentrations.
        Since exp_concs can contain np.nan, the total number of concentrations
        needs to be specified if dof-dependent statistics are to be returned.
        The error function can be modified but defaults to a simple difference
        (i.e., standard residual)

        """
        results = scipy.optimize.least_squares(self._residual,
                                self.ks_guesses + self.starting_concs_guesses, 
                                bounds=self.bounds,
                                args=(exp_times, exp_concs, error_function, monitor))

        fit_ks = list(results['x'][:self.num_ks])
        fit_concs = list(results['x'][self.num_ks:])

        reg_info = {}
        reg_info['sse'] = sum(results['fun']*results['fun'])
        if total_points:
            reg_info['total_points'] = total_points
            reg_info['dof'] = total_points - self.num_params
            reg_info['sde'] = reg_info['sse']/reg_info['dof']
        reg_info['success'] = results['success']
        reg_info['message'] = results['message']

        return fit_ks, fit_concs, reg_info

    def _solved_kin_sys(self, starting_concs, ks, times):
        """Solves the system of differential equations for given values
        of the parameters (ks) and starting concentrations (starting_concs)
        for a given set of time points (times).

        """
        return scipy.integrate.odeint(self.kin_sys, starting_concs, times, 
            args=tuple(ks), mxstep=self.MAX_STEPS)


    def _residual(self, parameters, exp_times, exp_concs, error_function, monitor=False): 
        """Calculates the residuals (as a np array) at a given time for a
        given set of parameters, passed as a list.

        The list of parameters (parameters) begins with the k's and K's
        and ends with the starting concentrations. The list of constants
        (consts) represents initial concentrations that will not be
        optimized.

        """
        ks = parameters[:self.num_ks]
        start_concs = np.append(parameters[self.num_ks:], self.starting_concs_constant)

        calcd_concs = self._solved_kin_sys(start_concs, ks, exp_times)

        residuals = [] # List of residuals.
        for n,r in itertools.product(range(len(exp_times)), range(self.num_concs)):
            # Ignores values for which there is no experimental data point.

            if not np.isnan(exp_concs[n][r]):
                residuals.append(error_function(float(exp_concs[n][r]), 
                                                calcd_concs[n][r]))

        # Print the sum of squares of the residuals to stdout, for the
        # purpose of monitoring progress.
        if monitor:
            print(sum([n**2 for n in residuals]))
        
        return np.array(residuals)
