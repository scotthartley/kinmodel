"""Defines the KineticModel class.

This class is used for kinetic models that can be fit to experimental
data or simulated with a given set of parameters.

"""
import itertools
import scipy.integrate, scipy.optimize, scipy.linalg, numpy as np
from .Dataset import Dataset

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
        starting_concs_constant (list of floats): Starting concentrations 
            that will be held constant.
        parameter_names (list of strings): Labels used for the
            parameters.
        legend_names (list of strings): Labels used for the
            concentrations in plots.
        top_plot (list of ints): Concentrations that should be plotted
            on top plot of output.
        bottom_plot (list of ints): (see top_plot)
        sort_order (list of ints): Translates order of columns in 
            experimental input into order of concentrations as defined 
            in model.
        int_eqn (list of functions): Functions of concentration and
            parameters that will be integrated, typically passed
            as lambda functions of lists of concentrations and 
            parameters (e.g., lambda cs, ks: ...)
        int_eqn_desc (list of strings): Descriptions of int_eqn.
        weight_func: Weighting function for residuals.
        bounds (tuple of floats, default (0, np.inf)): bounds for 
            parameters.

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
                 weight_func = lambda exp: 1,
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
        self.weight_func = weight_func
        self.bounds = bounds

    @property
    def num_concs(self):
        return (len(self.starting_concs_guesses) 
                + len(self.starting_concs_constant))

    @property
    def num_var_concs(self):
        return len(self.starting_concs_guesses)

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
        for a given number of points, maximum time, and with optional
        integration.

        """
        smooth_ts_out, deltaT = np.linspace(0, max_time, num_points, retstep=True)
        smooth_curves_out = self._solved_kin_sys(np.append(concs,
                self.starting_concs_constant), ks, smooth_ts_out)

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

    def fit_to_model(self, datasets, N_boot=0, monitor=False):
        """Performs a fit to a set of datasets containing time and
        concentration data.

        """
        num_datasets = len(datasets)
        total_points = sum(d.total_data_points for d in datasets)

        parameter_guesses = (self.ks_guesses 
                + self.starting_concs_guesses*num_datasets)

        results = scipy.optimize.least_squares(self._residual, 
                parameter_guesses, bounds=self.bounds, args=(datasets, monitor))

        reg_info = {}
        reg_info['dataset_names'] = [d.name for d in datasets]
        reg_info['dataset_times'] = [d.times.tolist() for d in datasets]
        reg_info['dataset_concs'] = [d.concs.tolist() for d in datasets]
        reg_info['num_datasets'] = num_datasets
        
        reg_info['fit_ks'] = list(results['x'][:self.num_ks])
        
        fit_concs = list(results['x'][self.num_ks:])
        reg_info['fit_concs'] = []
        for n in range(num_datasets):
            reg_info['fit_concs'].append([])
            for m in range(self.num_var_concs):
                reg_info['fit_concs'][-1].append(fit_concs[n*self.num_var_concs+m])

        # Residuals with weighting removed.
        pure_residuals = []
        for d in range(len(datasets)):
            solution = self._solved_kin_sys(
                    reg_info['fit_concs'][d] + self.starting_concs_constant,
                    reg_info['fit_ks'],
                    datasets[d].times)
            for n, m in itertools.product(range(datasets[d].num_times), range(self.num_concs)):
                if not np.isnan(datasets[d].concs[n,m]):
                    pure_residuals.append(datasets[d].concs[n,m] - solution[n,m])

        reg_info['success'] = results['success']
        reg_info['message'] = results['message']
        reg_info['ssr'] = results.cost * 2

        reg_info['total_points'] = total_points
        reg_info['total_params'] = self.num_ks + self.num_var_concs*num_datasets
        reg_info['dof'] = total_points - reg_info['total_params']
        reg_info['m_ssq'] = reg_info['ssr']/reg_info['dof']
        reg_info['rmsd'] = reg_info['m_ssq']**0.5

        reg_info['pure_ssr'] = sum(x**2 for x in pure_residuals)
        reg_info['pure_m_ssq'] = reg_info['pure_ssr']/reg_info['dof']
        reg_info['pure_rmsd'] = reg_info['pure_m_ssq']**0.5

        # See scipy.curve_fit; should yield same covariance matrix pcov.
        # (https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739)
        _, s, VT = scipy.linalg.svd(results.jac, full_matrices=False)
        reg_info['pcov'] = np.dot(VT.T / s**2, VT) * reg_info['m_ssq']
        reg_info['cov_errors'] = np.diag(reg_info['pcov'])**0.5
        # D_inv = Inverse sqrt of diagonal of covariance matrix,
        # used to calculate the correlation matrix.
        D_inv = np.linalg.inv(np.sqrt(np.diag(np.diag(reg_info['pcov']))))
        reg_info['corr'] = D_inv @ reg_info['pcov'] @ D_inv

        # Calculates parameter errors from bootstrap method. A random
        # residual is added to each data point and the resulting
        # "boot_dataset" is refit. Errors are obtained from the standard
        # deviations of the resulting parameter sets. Only executed if
        # N_boot >= 2 (makes no sense otherwise), should be much larger.
        if N_boot > 1:

            # Will store fit parameters for each bootstrap cycle.
            boot_params = np.empty((0, 
                    self.num_ks+self.num_var_concs*num_datasets))
            for i in range(N_boot):
                if monitor:
                    print(f"Bootstrapping iteration {i+1} of {N_boot}")
                
                # Perturbed datasets that can be fit.
                boot_datasets = []
                for d in datasets:
                    new_concs = np.empty((0, self.num_concs))
                    new_concs_t0 = []
                    # Leave starting conc 0's as 0's. These are not 
                    # unknowns.
                    for conc in d.concs[0]:
                        if not np.isnan(conc) and conc != 0:
                            new_concs_t0.append(conc 
                                    + np.random.choice(results['fun'])/self.weight_func(conc))
                        else:
                            new_concs_t0.append(conc)
                    new_concs = np.append(new_concs, [new_concs_t0], axis=0)

                    for n in range(1, d.num_times):
                        new_concs_t = []
                        for conc in d.concs[n]:
                            if not np.isnan(conc):
                                new_concs_t.append(conc 
                                        + np.random.choice(results['fun'])/self.weight_func(conc))
                            else:
                                new_concs_t.append(np.nan)
                        new_concs = np.append(new_concs, [np.array(new_concs_t)], 
                                axis=0)

                    boot_datasets.append(Dataset(times=d.times, concs=new_concs))

                boot_results = scipy.optimize.least_squares(self._residual,
                        results['x'], bounds=self.bounds, args=(boot_datasets, False))
                boot_params = np.append(boot_params, [boot_results['x']], axis=0)

                boot_std = []
                for n in range(self.num_ks + self.num_var_concs*num_datasets):
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


    def _residual(self, parameters, datasets, monitor=False): 
        """Calculates the residuals (as a np array) at a given time for
        a given set of parameters, passed as a list.

        Uses the error_function function. The list of parameters
        (parameters) begins with the k's and K's and ends with the
        starting concentrations. The list of constants (consts)
        represents initial concentrations that will not be optimized.

        """
        ks = parameters[:self.num_ks]

        num_datasets = len(datasets)
        var_concs_list = parameters[self.num_ks:]
        var_concs_per_dataset = len(var_concs_list)//num_datasets
        assert len(var_concs_list) % num_datasets == 0

        var_concs = []
        for n in range(num_datasets):
            current_concs = []
            for m in range(var_concs_per_dataset):
                current_concs.append(var_concs_list[n*var_concs_per_dataset+m])
            var_concs.append(current_concs)

        residuals = [] # List of residuals.
        for d in range(num_datasets):
            start_concs = np.append(var_concs[d], self.starting_concs_constant)
            calcd_concs = self._solved_kin_sys(start_concs, ks, 
                    datasets[d].times)

            for n,r in itertools.product(range(datasets[d].num_times), 
                    range(self.num_concs)):
                # Ignores values for which there is no experimental data
                # point. Will do all exp concs for a given time point
                # before moving on to next time.
                if not np.isnan(datasets[d].concs[n,r]):
                    residuals.append((datasets[d].concs[n,r] - calcd_concs[n,r])
                            *self.weight_func(datasets[d].concs[n,r]))

        # Print the sum of squares of the residuals to stdout, for the
        # purpose of monitoring progress.
        if monitor:
            print(f"Current ssr (weighted): {sum([n**2 for n in residuals])}")
        
        return np.array(residuals)
