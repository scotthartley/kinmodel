"""Defines the KineticModel class.

This class is used for kinetic models that can be fit to experimental
data or simulated with a given set of parameters.

"""
import sys
import os
import itertools
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.linalg
import math
from pathos.pools import ProcessPool
from pathos.helpers import mp as multiprocess
from .Dataset import Dataset
from .default_models import default_models

INDIRECT_DESC_SPACER = "\n\nOriginal model:\n"

np.set_printoptions(threshold=sys.maxsize)


class KineticModel:
    """Defines a kinetic model (as a system of differential equations).

    Attributes:
        name (string): Short form name used to identify model.
        description (string): Multiline description of model.
        kin_sys (list of equations): List of differential equations that
            will be solved.
        ks_guesses (list of floats): Initial guesses for k's and K's.
        ks_constant (list of floats): k's and K's that are constants.
        conc0_guesses (list of floats): Initial guesses for
            starting concs to be optimized.
        conc0_constant (list of floats): Starting concentrations
            that will be held constant.
        k_var_names, etc. (list of strings): Labels used for the
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

    MAX_STEPS = 10000  # Maximum steps in scipy.integrate.odeint

    def __init__(self,
                 name,
                 description,
                 kin_sys,
                 ks_guesses,
                 ks_constant,
                 conc0_guesses,
                 conc0_constant,
                 k_var_names,
                 k_const_names,
                 conc0_var_names,
                 conc0_const_names,
                 legend_names,
                 top_plot,
                 bottom_plot,
                 sort_order,
                 int_eqn=[],
                 int_eqn_desc=[],
                 weight_func=lambda exp: 1,
                 bounds=(0, np.inf),
                 lifetime_conc=[],
                 lifetime_fracs=[1, 1/2.71828, 0.1, 0.01],
                 rectime_conc=[],
                 rectime_fracs=[0.99],
                 ):

        self.name = name
        self.description = description
        self.kin_sys = kin_sys
        self.ks_guesses = ks_guesses
        self.ks_constant = ks_constant
        self.conc0_guesses = conc0_guesses
        self.conc0_constant = conc0_constant
        self.k_var_names = k_var_names
        self.k_const_names = k_const_names
        self.conc0_var_names = conc0_var_names
        self.conc0_const_names = conc0_const_names
        self.legend_names = legend_names
        self.top_plot = top_plot
        self.bottom_plot = bottom_plot
        self.sort_order = sort_order
        self.int_eqn = int_eqn
        self.int_eqn_desc = int_eqn_desc
        self.weight_func = weight_func
        self.bounds = bounds
        self.lifetime_conc = lifetime_conc
        self.lifetime_fracs = lifetime_fracs
        self.rectime_conc = rectime_conc
        self.rectime_fracs = rectime_fracs

    @property
    def num_concs0(self):
        return self.num_var_concs0 + self.num_const_concs0

    @property
    def num_data_concs(self):
        return self.num_concs0

    @property
    def num_var_concs0(self):
        return len(self.conc0_guesses)

    @property
    def num_const_concs0(self):
        return len(self.conc0_constant)

    @property
    def num_ks(self):
        return self.num_var_ks + self.num_const_ks

    @property
    def num_var_ks(self):
        return len(self.ks_guesses)

    @property
    def num_const_ks(self):
        return len(self.ks_constant)

    @property
    def num_params(self):
        return self.num_var_ks + self.num_var_concs0

    @property
    def num_consts(self):
        return self.num_const_ks + self.num_const_concs0

    @property
    def parameter_names(self):
        return list(self.k_var_names + self.conc0_var_names)

    @property
    def constant_names(self):
        return list(self.k_const_names + self.conc0_const_names)

    @property
    def len_params(self):
        return max(len(n) for n in self.parameter_names)

    @property
    def len_consts(self):
        return max(len(n) for n in self.constant_names)

    @property
    def len_legend(self):
        return max(len(n) for n in self.legend_names)

    @property
    def len_int_eqn_desc(self):
        return max(len(n) for n in self.int_eqn_desc)

    def simulate(self, ks, concs, num_points, max_time, integrate=True):
        """Using _solved_kin_sys, solves the system of diff. equations
        for a given number of points, maximum time, and with optional
        integration.

        Both ks and concs can be either just the variable values or the
        all possible values.

        """
        smooth_ts_out, deltaT = np.linspace(0, max_time, num_points,
                                            retstep=True)

        if len(ks) == self.num_var_ks:
            all_ks = np.append(ks, self.ks_constant)
        elif len(ks) == self.num_ks:
            all_ks = ks
        else:
            raise(RuntimeError(f"Invalid number of k's specified: {ks}"))

        if len(concs) == self.num_var_concs0:
            all_concs = np.append(concs, self.conc0_constant)
        elif len(concs) == self.num_concs0:
            all_concs = concs
        else:
            raise(RuntimeError("Invalid number of concentrations specified."))

        smooth_curves_out = self._solved_kin_sys(all_concs, all_ks,
                                                 smooth_ts_out)

        if integrate:
            integrals = {}
            for i in range(len(self.int_eqn)):
                integral_func = []
                for t in smooth_curves_out:
                    integral_func.append(self.int_eqn[i](t, ks))
                integrals[self.int_eqn_desc[i]] = scipy.integrate.simps(
                        integral_func, dx=deltaT)
        else:
            integrals = None

        return smooth_ts_out, smooth_curves_out, integrals

    def fit_to_model(self, datasets, ks_guesses=None,
                     conc0_guesses=None, ks_const=None,
                     conc0_const=None, N_boot=0, monitor=False,
                     boot_CI=95, boot_points=1000, boot_t_exp=1.1,
                     boot_force1st=False, boot_nodes=None):
        """Performs a fit to a set of datasets containing time and
        concentration data.

        """
        num_datasets = len(datasets)
        total_points = sum(d.total_data_points for d in datasets)

        parameter_guesses = []
        if ks_guesses:
            if len(ks_guesses) == self.num_var_ks:
                parameter_guesses += ks_guesses
            else:
                raise(RuntimeError("Invalid number of k's specified"))
        else:
            parameter_guesses += self.ks_guesses
        if conc0_guesses:
            if len(conc0_guesses) == self.num_var_concs0:
                parameter_guesses += conc0_guesses*num_datasets
            elif len(conc0_guesses) == self.num_var_concs0*num_datasets:
                parameter_guesses += conc0_guesses
            else:
                raise(RuntimeError(
                        "Invalid number of concentrations specified."))
        else:
            parameter_guesses += self.conc0_guesses*num_datasets

        parameter_constants = []
        if ks_const:
            if len(ks_const) == self.num_const_ks:
                parameter_constants += ks_const
            else:
                raise(RuntimeError("Invalid number of k's specified."))
        else:
            parameter_constants += self.ks_constant
        if conc0_const:
            if len(conc0_const) == self.num_const_concs0:
                parameter_constants += conc0_const*num_datasets
            elif len(conc0_const) == self.num_const_concs0*num_datasets:
                parameter_constants += conc0_const
            else:
                raise(RuntimeError(
                        "Invalid number of concentrations specified."))
        else:
            parameter_constants += self.conc0_constant*num_datasets

        results = scipy.optimize.least_squares(
                self._residual, parameter_guesses, bounds=self.bounds,
                args=(datasets, parameter_constants, monitor))

        reg_info = {}
        reg_info['dataset_names'] = [d.name for d in datasets]
        reg_info['dataset_times'] = [d.times.tolist() for d in datasets]
        reg_info['dataset_concs'] = [d.concs.tolist() for d in datasets]
        reg_info['num_datasets'] = num_datasets

        max_exp_concs = [0]*self.num_data_concs
        for d in datasets:
            for n in range(self.num_data_concs):
                if not np.all(np.isnan(d.concs[:, n])):
                    if np.nanmax(d.concs[:, n]) > max_exp_concs[n]:
                        max_exp_concs[n] = np.nanmax(d.concs[:, n])
        reg_info['max_exp_concs'] = max_exp_concs

        reg_info['fit_ks'] = list(results['x'][:self.num_var_ks])
        fit_concs = list(results['x'][self.num_var_ks:])
        reg_info['fit_concs'] = []
        for n in range(num_datasets):
            reg_info['fit_concs'].append(self._dataset_concs(
                    n, fit_concs, self.num_var_concs0))
        reg_info['fixed_ks'] = list(parameter_constants[:self.num_const_ks])
        fixed_concs = list(parameter_constants[self.num_const_ks:])
        reg_info['fixed_concs'] = []
        for n in range(num_datasets):
            reg_info['fixed_concs'].append(
                    self._dataset_concs(n, fixed_concs, self.num_const_concs0))

        reg_info['predicted_data'] = []
        for d in range(num_datasets):
            reg_info['predicted_data'].append(self._solved_kin_sys(
                    reg_info['fit_concs'][d] + reg_info['fixed_concs'][d],
                    reg_info['fit_ks'] + reg_info['fixed_ks'],
                    reg_info['dataset_times'][d]))

        max_pred_concs = [0]*self.num_data_concs
        for d in reg_info['predicted_data']:
            for n in range(self.num_data_concs):
                if np.nanmax(d[:, n]) > max_pred_concs[n]:
                    max_pred_concs[n] = np.nanmax(d[:, n])
        reg_info['max_pred_concs'] = max_pred_concs

        reg_info['success'] = results['success']
        reg_info['message'] = results['message']
        reg_info['ssr'] = results.cost * 2

        reg_info['total_points'] = total_points
        reg_info['total_params'] = (self.num_ks
                                    + self.num_var_concs0*num_datasets)
        reg_info['dof'] = total_points - reg_info['total_params']
        reg_info['m_ssq'] = reg_info['ssr']/reg_info['dof']
        reg_info['rmsd'] = reg_info['m_ssq']**0.5

        pure_residuals = self._pure_residuals(datasets, reg_info,
                                              parameter_constants)

        # reg_info['pure_ssr'] = np.nansum(np.square(pure_residuals))
        reg_info['pure_ssr'] = sum([np.nansum(np.square(r)) for r in pure_residuals])
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

        if N_boot > 1:
            reg_info['boot_num'] = N_boot

            reg_info['boot_method'] = "random-X"
            reg_info['boot_force1st'] = True if boot_force1st else False
            all_boot_datasets = Dataset.boot_randomX(
                                        N_boot, datasets,
                                        force1st=boot_force1st)

            reg_info['boot_fit_ks'], reg_info['boot_fit_concs'] = (
                    self.bootstrap(
                            all_boot_datasets, results['x'],
                            parameter_constants, monitor, nodes=boot_nodes))
            reg_info['boot_param_CIs'] = []
            reg_info['boot_plot_CIs'] = []
            reg_info['boot_plot_ts'] = []
            for d in range(num_datasets):
                if monitor:
                    print(f"Simulating bootstrap dataset {d+1} of "
                          f"{num_datasets}")
                reg_info['boot_param_CIs'].append(self.bootstrap_param_CIs(
                        reg_info, d, boot_CI))
                boot_CIs, boot_ts = self.bootstrap_plot_CIs(
                        reg_info, d, boot_CI, boot_points, boot_t_exp, monitor)
                reg_info['boot_plot_CIs'].append(boot_CIs)
                reg_info['boot_plot_ts'].append(boot_ts)

        return reg_info

    def _pure_residuals(self, datasets, reg_info, parameter_constants):
        """Returns unweighted residuals.
        """
        residuals = []
        for d in range(len(datasets)):
            dataset_residuals = []
            d_ks = np.append(
                    reg_info['fit_ks'],
                    parameter_constants[:self.num_const_ks])
            d_concs = np.append(
                    reg_info['fit_concs'][d],
                    self._dataset_concs(
                            d, parameter_constants[self.num_const_ks:],
                            self.num_const_concs0))
            solution = self._solved_kin_sys(d_concs, d_ks, datasets[d].times)
            for m in range(self.num_data_concs):
                dataset_residuals.append([])
                for n in range(datasets[d].num_times):
                    if not np.isnan(datasets[d].concs[n, m]):
                        dataset_residuals[m].append(
                                datasets[d].concs[n, m] - solution[n, m])
                    else:
                        dataset_residuals[m].append(np.nan)
            residuals.append(np.array(dataset_residuals))

        return residuals

    def bootstrap(self, all_datasets, fit_params, constants,
                  monitor=False, nodes=None):
        """Process a set of datasets obtained by a bootstrapping method,
        returning the ks and concs.

        """
        def _results(inp):
            """Used to parallelize fitting of datasets.

            """
            datasets, cnt, lock = inp
            if monitor:
                with lock:
                    cnt.value += 1
                    print(f"Bootstrapping fit {cnt.value} of {total_datasets}")
            fit = scipy.optimize.least_squares(
                    self._residual, fit_params, bounds=self.bounds,
                    args=(datasets, constants, False))['x']
            return fit

        total_datasets = len(all_datasets)
        num_datasets = len(all_datasets[0])
        boot_params = np.empty(
                (0, self.num_var_ks+self.num_var_concs0*num_datasets))

        # # Old code that runs bootstrap fits in serial.
        # n = 0
        # for datasets in all_datasets:
        #     n += 1
        #     if monitor:
        #         print(f"Bootstrapping fit {n} of {total_datasets}")
        #     boot_params = np.append(boot_params, [results(datasets, fit_params, constants)], axis=0)

        # New code that runs bootstrap fits in parallel.

        with ProcessPool(nodes=nodes) as p:
            lock = multiprocess.Manager().Lock()
            counter = multiprocess.Manager().Value('i', 0)
            boot_params = p.map(
                _results, [(d, counter, lock) for d in all_datasets])

        boot_fit_ks = []
        boot_fit_concs = [[] for _ in range(num_datasets)]
        for p in boot_params:
            boot_fit_ks.append(list(p[:self.num_var_ks]))

            for n in range(num_datasets):
                boot_fit_concs[n].append(self._dataset_concs(
                        n, list(p[self.num_var_ks:]), self.num_var_concs0))

        return np.array(boot_fit_ks), np.array(boot_fit_concs)

    def bootstrap_param_CIs(self, reg_info, dataset_n, CI):
        """Returns upper and lower confidence intervals for bootstrapped
        statistics, as an ndarray.
        """
        k_CIs = np.percentile(
                    reg_info['boot_fit_ks'], [(100-CI)/2, (100+CI)/2], axis=0)
        conc_CIs = np.percentile(
                reg_info['boot_fit_concs'][dataset_n],
                [(100-CI)/2, (100+CI)/2], axis=0)

        return k_CIs, conc_CIs

    def bootstrap_plot_CIs(self, reg_info, dataset_n, CI, num_points,
                           time_exp_factor, monitor=False):
        """Returns upper and lower confidence intervals for bootstrapped
        statistics, as an ndarray.
        """
        max_time = max(reg_info['dataset_times'][dataset_n])*time_exp_factor
        smooth_ts_out, _ = np.linspace(0, max_time, num_points, retstep=True)

        boot_iterations = reg_info['boot_num']
        CI_num = math.ceil(boot_iterations*(100-CI)/200)
        assert CI_num > 0

        CI_top = np.empty((0, num_points, self.num_data_concs))
        CI_bot = np.empty((0, num_points, self.num_data_concs))
        for n in range(boot_iterations):
            if monitor:
                print(f"Bootstrapping simulation {n+1} "
                      f"of {reg_info['boot_num']}", end="")
            _, boot_plot, _ = self.simulate(
                    reg_info['boot_fit_ks'][n],
                    reg_info['boot_fit_concs'][dataset_n][n], num_points,
                    max_time, integrate=False)
            if (n+1) <= CI_num:
                CI_top = np.append(CI_top, [boot_plot], axis=0)
                CI_top = np.sort(CI_top, axis=0)
                CI_bot = np.append(CI_bot, [boot_plot], axis=0)
                CI_bot = np.sort(CI_bot, axis=0)
            else:
                if not np.all(np.maximum(CI_top[0], boot_plot) == CI_top[0]):
                    if monitor:
                        print(', top CI increased', end="")
                    CI_top = np.append(CI_top, [boot_plot], axis=0)
                    CI_top = np.sort(CI_top, axis=0)
                    CI_top = CI_top[1:]
                if not np.all(np.minimum(CI_bot[-1], boot_plot) == CI_bot[-1]):
                    if monitor:
                        print(', bottom CI decreased', end="")
                    CI_bot = np.append(CI_bot, [boot_plot], axis=0)
                    CI_bot = np.sort(CI_bot, axis=0)
                    CI_bot = CI_bot[:-1]
            if monitor:
                print()

        return (CI_top[0], CI_bot[-1]), smooth_ts_out

    def _solved_kin_sys(self, conc0, ks, times):
        """Solves the system of differential equations for given values
        of the parameters (ks) and starting concentrations (conc0)
        for a given set of time points (times).

        """
        return scipy.integrate.odeint(
                self.kin_sys, conc0, times, args=tuple(ks),
                mxstep=self.MAX_STEPS)

    def _residual(self, parameters, datasets, constants, monitor=False):
        """Calculates the residuals (as a np array) at a given time for
        a given set of parameters, passed as a list.

        Uses the error_function function. The list of parameters
        (parameters) begins with the k's and K's and ends with the
        starting concentrations. The list of constants (consts)
        represents initial concentrations that will not be optimized.

        """
        num_datasets = len(datasets)

        ks_var = parameters[:self.num_var_ks]
        var_concs0_list = parameters[self.num_var_ks:]
        var_concs0_per_dataset = len(var_concs0_list)//num_datasets
        assert len(var_concs0_list) % num_datasets == 0
        var_concs0 = []
        for n in range(num_datasets):
            var_concs0.append(self._dataset_concs(n, var_concs0_list,
                              var_concs0_per_dataset))

        ks_const = constants[:self.num_const_ks]
        const_concs0_list = constants[self.num_const_ks:]
        const_concs0_per_dataset = len(const_concs0_list)//num_datasets
        assert len(const_concs0_list) % num_datasets == 0
        const_concs0 = []
        for n in range(num_datasets):
            const_concs0.append(self._dataset_concs(n, const_concs0_list,
                                const_concs0_per_dataset))

        all_ks = np.append(ks_var, ks_const)

        residuals = []  # List of residuals.
        for d in range(num_datasets):
            start_concs0 = np.append(var_concs0[d], const_concs0[d])
            calcd_concs = self._solved_kin_sys(start_concs0, all_ks,
                                               datasets[d].times)

            for n, r in itertools.product(range(datasets[d].num_times),
                                          range(self.num_data_concs)):
                # Ignores values for which there is no experimental data
                # point. Will do all exp concs for a given time point
                # before moving on to next time.
                if not np.isnan(datasets[d].concs[n, r]):
                    residuals.append(
                            (datasets[d].concs[n, r] - calcd_concs[n, r])
                            * self.weight_func(datasets[d].concs[n, r]))

        # Print the sum of squares of the residuals to stdout, for the
        # purpose of monitoring progress.
        if monitor:
            print(f"Current ssr (weighted): {sum([n**2 for n in residuals])}")

        return np.array(residuals)

    @staticmethod
    def _dataset_concs(n, c_list, c_per_n):
        """Returns the concentrations corresponding to a specific
        dataset n from a list of concs for all datasets.

        """

        return c_list[(n*c_per_n):(n*c_per_n + c_per_n)]

    @staticmethod
    def get_model(model_name, new_model=None):
        """Returns the model object corresponding to model_name, with
        the option of specifying an additional file with new models.
        """
        if new_model:
            sys.path.append(os.getcwd())
            new_model = __import__(new_model).model
            models = {**default_models, **{new_model.name: new_model}}
        else:
            models = default_models

        try:
            return models[model_name]
        except KeyError:
            print(f'"{model_name}" is not a valid model.')
            print(", ".join(a for a in models), "are currently available.")
            sys.exit(1)


class IndirectKineticModel(KineticModel):
    """Defines an indirect kinetic model: one in which the observed data
    does not map cleanly onto species' concentrations.

    """

    def __init__(self,
                 name,
                 parent_model_name,
                 description,
                 conc_mapping,
                 legend_names,
                 top_plot,
                 bottom_plot,
                 sort_order,
                 int_eqn=[],
                 int_eqn_desc=[],
                 lifetime_conc=[],
                 lifetime_fracs=[1, 1/2.71828, 0.1, 0.01],
                 rectime_conc=[],
                 rectime_fracs=[0.99],
                 ):

        self.parent_model = KineticModel.get_model(parent_model_name)
        self.conc_mapping = conc_mapping

        self.name = name
        self.kin_sys = self.parent_model.kin_sys
        self.description = (description
                            + INDIRECT_DESC_SPACER
                            + self.parent_model.description)
        self.ks_guesses = self.parent_model.ks_guesses
        self.ks_constant = self.parent_model.ks_constant
        self.conc0_guesses = self.parent_model.conc0_guesses
        self.conc0_constant = self.parent_model.conc0_constant
        self.k_var_names = self.parent_model.k_var_names
        self.k_const_names = self.parent_model.k_const_names
        self.conc0_var_names = self.parent_model.conc0_var_names
        self.conc0_const_names = self.parent_model.conc0_const_names
        self.legend_names = legend_names
        self.top_plot = top_plot
        self.bottom_plot = bottom_plot
        self.sort_order = sort_order
        self.int_eqn = int_eqn
        self.int_eqn_desc = int_eqn_desc
        self.weight_func = self.parent_model.weight_func
        self.bounds = self.parent_model.bounds
        self.lifetime_conc = lifetime_conc
        self.lifetime_fracs = lifetime_fracs
        self.rectime_conc = rectime_conc
        self.rectime_fracs = rectime_fracs

    @property
    def num_data_concs(self):
        return len(self.legend_names)

    def _solved_kin_sys(self, conc0, ks, times):
        parent_model_soln = scipy.integrate.odeint(
                self.kin_sys, conc0, times, args=tuple(ks),
                mxstep=self.MAX_STEPS)

        return self.conc_mapping(parent_model_soln)
