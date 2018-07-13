"""Defines the KineticModel class.

This class is used for kinetic models that can be fit to experimental
data or simulated with a given set of parameters.

"""
import itertools
import scipy.integrate, scipy.optimize, scipy.linalg, numpy as np

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
        error_function: The error function to be used in regression. By
            default, just the difference between the experimental and
            calculated values (normal residual for least sq regression).
        undo_error_function: The function to convert a residual back to a
            meaningful error for a given experimental concentration. 
            Necessary is more complex error functions are to be used in
            combination with the bootstrap error estimation on the
            parameters.
        add_residual: Also needed for bootstrap error estimation if
            more complex error function used.

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
                 error_function=lambda exp, calc: exp - calc,
                 undo_error_function=lambda exp, res: res,
                 add_residual=lambda exp, res: exp + res,
                 bounds = (0, np.inf),
                 ):
        
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
        self.error_function = error_function
        self.undo_error_function = undo_error_function
        self.add_residual = add_residual
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
        """Using _solved_kin_sys, solves the system of diff. equations
        for a with a given number of points, maximum time, and with
        optional integration.

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

    def fit_to_model(self, exp_times, exp_concs, N_boot=0, monitor=False):
        """Performs a fit to a set of experimental times and concentrations.
        
        Since exp_concs can contain np.nan, the total number of concentrations
        needs to be specified if dof-dependent statistics are to be returned.
        The error function can be modified but defaults to a simple difference
        (i.e., standard residual)

        """
        total_points = exp_concs.size - np.isnan(exp_concs).sum()

        results = scipy.optimize.least_squares(self._residual,
                                self.ks_guesses + self.starting_concs_guesses, 
                                bounds=self.bounds,
                                args=(exp_times, exp_concs, monitor))

        reg_info = {}
        reg_info['fit_ks'] = list(results['x'][:self.num_ks])
        reg_info['fit_concs'] = list(results['x'][self.num_ks:])


        reg_info['success'] = results['success']
        reg_info['message'] = results['message']
        reg_info['ssr'] = results.cost * 2
        if total_points:
            reg_info['total_points'] = total_points
            reg_info['dof'] = total_points - self.num_params
            reg_info['m_ssq'] = reg_info['ssr']/reg_info['dof']
            reg_info['rmsd'] = reg_info['m_ssq']**0.5

            # See scipy.curve_fit; should yield same covariance matrix
            # pcov.
            # (https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739)
            _, s, VT = scipy.linalg.svd(results.jac, full_matrices=False)
            reg_info['pcov'] = np.dot(VT.T / s**2, VT) * reg_info['m_ssq']
            reg_info['cov_errors'] = np.diag(reg_info['pcov'])**0.5

            # D_inv = Inverse sqrt of diagonal of covariance matrix,
            # used to calculate the correlation matrix.
            D_inv = np.linalg.inv(np.sqrt(np.diag(np.diag(reg_info['pcov']))))
            reg_info['corr'] = D_inv @ reg_info['pcov'] @ D_inv
        
            # Calculates parameter errors from bootstrap method. A
            # random residual is added to each data point and the result
            # is refit. Errors are obtained from the resulting parameter
            # sets. Only executed if N_boot >= 2 (makes no sense
            # otherwise), should be much larger. Should accommodate more
            # complex error functions so long as the error_function,
            # undo_error_function, and add_residual methods have been
            # properly defined.
            if N_boot > 1:
                pure_residuals = [] # Residuals with scaling removed.
                for n in range(len(exp_times)):
                    for m in range(len(exp_concs[0])):
                        if not np.isnan(exp_concs[n,m]):
                            pure_residuals.append(
                                self.undo_error_function(exp_concs[n,m],
                                    results.fun[n+m]))

                boot_params = np.empty((0,self.num_params))
                for i in range(N_boot):
                    boot_concs = np.empty((0,len(exp_concs[0])))

                    # At t=0, don't permute concentrations that are 0 since
                    # these species cannot be present.
                    concs_t0 = []
                    for n in exp_concs[0]:
                        if not np.isnan(n) and n != 0:
                            concs_t0.append(self.add_residual(n, 
                                np.random.choice(pure_residuals)))
                        else:
                            concs_t0.append(n)
                    boot_concs = np.append(boot_concs, [concs_t0], axis=0)
                    # Loop through remaining times, adding concentrations and
                    # perturbing by residuals using the add_residual function.
                    for n in range(1,len(exp_times)):
                        concs_t = []
                        for m in range(len(exp_concs[0])):
                            if not np.isnan(exp_concs[n,m]):
                                concs_t.append(self.add_residual(exp_concs[n,m], 
                                    np.random.choice(pure_residuals)))
                            else:
                                concs_t.append(np.nan)
                        boot_concs = np.append(boot_concs, [np.array(concs_t)], 
                                               axis=0)

                    if monitor:
                        print(boot_concs)

                    boot_results = scipy.optimize.least_squares(self._residual,
                                       results['x'], bounds=self.bounds,
                                       args=(exp_times, boot_concs, False))
                    boot_params = np.append(boot_params, [boot_results['x']], 
                                            axis=0)

                # Store statistics for the parameters from the bootstrapping
                # fits.
                boot_std = []
                for n in range(self.num_params):
                    boot_std.append(np.std(boot_params[:,n]))
                reg_info['boot_stddevs'] = boot_std
                reg_info['boot_num'] = N_boot

        return reg_info

    def _solved_kin_sys(self, starting_concs, ks, times):
        """Solves the system of differential equations for given values
        of the parameters (ks) and starting concentrations (starting_concs)
        for a given set of time points (times).

        """
        return scipy.integrate.odeint(self.kin_sys, starting_concs, times, 
            args=tuple(ks), mxstep=self.MAX_STEPS)


    def _residual(self, parameters, exp_times, exp_concs, monitor=False): 
        """Calculates the residuals (as a np array) at a given time for a
        given set of parameters, passed as a list. 

        Uses the error_function function. The list of parameters
        (parameters) begins with the k's and K's and ends with the
        starting concentrations. The list of constants (consts)
        represents initial concentrations that will not be optimized.

        """
        ks = parameters[:self.num_ks]
        start_concs = np.append(parameters[self.num_ks:], self.starting_concs_constant)

        calcd_concs = self._solved_kin_sys(start_concs, ks, exp_times)

        residuals = [] # List of residuals.
        for n,r in itertools.product(range(len(exp_times)), range(self.num_concs)):
            # Ignores values for which there is no experimental data
            # point. Will do all exp concs for a given time point before
            # moving on to next time.
            if not np.isnan(exp_concs[n][r]):
                residuals.append(self.error_function(float(exp_concs[n][r]), 
                                                calcd_concs[n][r]))

        # Print the sum of squares of the residuals to stdout, for the
        # purpose of monitoring progress.
        if monitor:
            print(sum([n**2 for n in residuals]))
        
        return np.array(residuals)
