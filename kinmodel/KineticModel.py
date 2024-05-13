"""Defines the KineticModel class.

This class is used for kinetic models that can be fit to experimental
data or simulated with a given set of parameters.


** Modified by Gyunam Park 24.02.28

"""
import sys
import os
import itertools
import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.linalg
import math
from joblib import Parallel, delayed
from tqdm import tqdm
from .Dataset import Dataset
import yaml
import kinmodel.constants as constants
import copy

INDIRECT_DESC_SPACER = "\n\nOriginal model:\n"
INT_LAMBDA = "lambda c, k: "
CALC_LAMBDA = "lambda c, t, k, i: "
CONC_MAP_LAMBDA = ["lambda c: np.array([", "]).transpose()"]

np.set_printoptions(threshold=sys.maxsize)


class KineticModel:
    """Defines a kinetic model (as a system of differential equations).

    Attributes:
        name (string): Short form name used to identify model.
        description (string): Multiline description of model.
        eq_function (string): A string that will be converted into a
            function defining the kinetic system (using exec).
        k_var, k_const: Dictionaries of parameters (variable or
            constant) with name and guess/value fields.
        conc0_var, conc0_const: Dictionaries of initial concentrations
            (variable or constant) with name and guess/value fields.
        species: List of concentrations that will be plotted,
            with name, plot ("top" or "bottom"), and sort fields.
            "sort" defines the order of columns in input files.
        integrals: List of dictionaries that will be integrated over the
            plot. Each includes a desc (description) and func
            (function). The function must be of parameters (k)
            and concentrations (c), indexed as lists (e.g., c[0]).
        calcs: Similar to integrals. Quantities that will be calculated
            from the data. Should be functions of concentrations (c),
            time (t), parameters (k), and integrals (i).
        lifetime_concs/fracs: Species for which lifetimes will be
            calculated, and what fractions will be used to define
            lifetimes. List of indices.
        rectime_concs/fracs: Similar to the lifetimes, but for recovery
            times.
        weight_func: Weighting function for residuals.
        bounds (tuple of floats, default (0, np.inf)): bounds for
            parameters.
        path: path to the original model definition file.

    """

    MAX_STEPS = 10000  # Maximum steps in scipy.integrate.odeint

    def __init__(self,
                 name,
                 eq_function,
                 species,
                 description=None,
                 k_var=[],
                 k_const=[],
                 conc0_var=[],
                 conc0_const=[],
                 integrals=[],
                 calcs=[],
                 lifetime_concs=[],
                 lifetime_fracs=[1, 1/2.71828, 0.1, 0.01],
                 rectime_concs=[],
                 rectime_fracs=[0.99],
                 weight_func=lambda exp: 1,
                 bounds=(0, np.inf),
                 path=None,
                 **kwargs
                 ):

        self.name = name
        self.description = description

        # Loads the "equations" function defined in the model.
        locals_dict = {}
        exec(eq_function, locals_dict)
        self.kin_sys = locals_dict['equations']

        self.k_var_names = [k['name'] for k in k_var] if k_var else []
        self.ks_guesses = [k['guess'] for k in k_var] if k_var else []
        self.k_const_names = [k['name'] for k in k_const] if k_const else []
        self.ks_constant = [k['value'] for k in k_const] if k_const else []
        self.conc0_var_names = (
                [c['name'] for c in conc0_var] if conc0_var else [])
        self.conc0_guesses = (
                [c['guess'] for c in conc0_var] if conc0_var else [])
        self.conc0_const_names = (
                [c['name'] for c in conc0_const] if conc0_const else [])
        self.conc0_constant = (
                [c['value'] for c in conc0_const] if conc0_const else [])

        self.legend_names = []
        self.sort_order = []
        self.top_plot = []
        self.bottom_plot = []
        for n in range(len(species)):
            self.legend_names.append(species[n]['name'])
            if 'sort' in species[n]:
                self.sort_order.append(species[n]['sort'])
            if species[n]['plot'] == "top":
                self.top_plot.append(n)
            elif species[n]['plot'] == "bottom":
                self.bottom_plot.append(n)

        self.int_eqn = []
        self.int_eqn_desc = []
        if integrals:
            for i in integrals:
                self.int_eqn.append(eval(INT_LAMBDA + i['func']))
                self.int_eqn_desc.append(i['desc'])

        self.calcs = []
        self.calcs_desc = []
        if calcs:
            for c in calcs:
                self.calcs.append(eval(CALC_LAMBDA + c['func']))
                self.calcs_desc.append(c['desc'])

        self.weight_func = weight_func
        self.bounds = bounds
        self.lifetime_conc = lifetime_concs
        self.lifetime_fracs = lifetime_fracs
        self.rectime_conc = rectime_concs
        self.rectime_fracs = rectime_fracs
        self.path = path

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
        if self.parameter_names:
            return max(len(n) for n in self.parameter_names)
        else:
            return 0

    @property
    def len_consts(self):
        if self.constant_names:
            return max(len(n) for n in self.constant_names)
        else:
            return 0

    @property
    def len_legend(self):
        if self.legend_names:
            return max(len(n) for n in self.legend_names)
        else:
            return 0

    @property
    def len_int_eqn_desc(self):
        if self.int_eqn_desc:
            return max(len(n) for n in self.int_eqn_desc)
        else:
            return 0

    @property
    def len_calcs_desc(self):
        if self.calcs_desc:
            return max(len(n) for n in self.calcs_desc)
        else:
            return 0

    @property
    def num_calcs(self):
        return len(self.calcs)

    def get_elements_of_nested_list(self,element): # new definition
        count = 0
        if isinstance(element, list):
            for each_element in element:
                count += self.get_elements_of_nested_list(each_element)
        else:
            count += 1
        return count

    def simulate(self, ks, concs, smooth_ts_out, integrate=False, # argument change
                 calcs=False):
        """Using _solved_kin_sys, solves the system of diff. equations
        for a given number of points, maximum time, and with optional
        integration.

        Both ks and concs can be either just the variable values or the
        all possible values.

        """
        #smooth_ts_out, deltaT = np.linspace(0, max_time, num_points, #comment this line
        #                                    retstep=True)        

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
            integrals = []
            for i in range(len(self.int_eqn)):
                integral_func = []
                for cs in smooth_curves_out:
                    integral_func.append(self.int_eqn[i](cs, all_ks))
                integrals.append(
                        (self.int_eqn_desc[i],
                         scipy.integrate.simps(integral_func, smooth_ts_out)))
        else:
            integrals = None

        if calcs:
            calc_results = []
            for i in range(self.num_calcs):
                calc_results.append(
                        (self.calcs_desc[i],
                         self.calcs[i](
                                smooth_curves_out.transpose(),
                                smooth_ts_out,
                                np.append(ks, concs),
                                [i[1] for i in integrals])))
        else:
            calc_results = None

        return smooth_ts_out, smooth_curves_out, integrals, calc_results

    def fit_to_model(self, datasets, ks_guesses=None,
                     conc0_guesses=None, ks_const=None,
                     conc0_const=None, N_boot=0, monitor=False,
                     boot_CI=95, boot_points=1000, boot_t_exp=1.1,
                     boot_force1st=False, boot_nodes=-1,
                     cc_ints=10, cc_mult=3.0, cc_include_cs=False, plot_semilogx=True): # New word
        """Performs a fit to a set of datasets containing time and
        concentration data.

        """
        def _sim_boot(inp):
            """Wrapper function to parallelize bootstrap simulations.
            """
            d = inp
            param_CIs = self.bootstrap_param_CIs(reg_info, d, boot_CI)
            boot_CIs, boot_calc_CIs, boot_ts = self.bootstrap_plot_CIs(
                    reg_info, d, boot_CI, boot_points, boot_t_exp,
                    monitor=False)
            return d, param_CIs, boot_CIs, boot_calc_CIs, boot_ts

        num_datasets = len(datasets)
        total_points = sum(d.total_data_points for d in datasets)

        parameter_guesses = []
        if ks_guesses:
            if len(ks_guesses) == self.num_var_ks:
                parameter_guesses += ks_guesses
            else:
                raise(RuntimeError(f"Invalid number of k's specified "
                                   f"({len(ks_guesses)} expected "
                                   f"{self.num_var_ks})"))
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
                args=(datasets, parameter_constants, False))

        reg_info = {}
        reg_info['datasets'] = datasets
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

        reg_info['all_params'] = results['x']
        reg_info['fit_ks'] = list(results['x'][:self.num_var_ks])
        print(reg_info['fit_ks'])
        fit_concs = list(results['x'][self.num_var_ks:])
        reg_info['fit_concs'] = []
        for n in range(num_datasets):
            reg_info['fit_concs'].append(self._dataset_concs(
                    n, fit_concs, self.num_var_concs0))
        reg_info['parameter_constants'] = parameter_constants
        reg_info['fixed_ks'] = list(parameter_constants[:self.num_const_ks])
        fixed_concs = list(parameter_constants[self.num_const_ks:])
        reg_info['fixed_concs'] = []
        for n in range(num_datasets):
            reg_info['fixed_concs'].append(
                    self._dataset_concs(n, fixed_concs, self.num_const_concs0))

        reg_info['predicted_data'] = []
        reg_info['predicted_time'] = [] # new line
        for d in range(num_datasets):
            if plot_semilogx:
                if datasets[d].min_time == 0:                     
                    predicted_time = np.concatenate((np.zeros(1,),np.logspace(np.log10(datasets[d].second_time), np.log10(datasets[d].max_time), boot_points-1)),axis=0)
                else:
                    predicted_time = np.logspace(np.log10(datasets[d].min_time), np.log10(datasets[d].max_time), boot_points)

            else:                    
                predicted_time = np.linspace(datasets[d].min_time, datasets[d].max_time, boot_points)

            reg_info['predicted_time'].append(predicted_time)
            reg_info['predicted_data'].append(self._solved_kin_sys(
                    reg_info['fit_concs'][d] + reg_info['fixed_concs'][d],
                    reg_info['fit_ks'] + reg_info['fixed_ks'],
                    predicted_time))

        max_pred_concs = [0]*self.num_data_concs
        for d in reg_info['predicted_data']:
            for n in range(self.num_data_concs):
                if np.nanmax(d[:, n]) > max_pred_concs[n]:
                    max_pred_concs[n] = np.nanmax(d[:, n])
        reg_info['max_pred_concs'] = max_pred_concs

        reg_info['success'] = results['success']
        reg_info['message'] = results['message']
        reg_info['ssr'] = results.cost * 2

        if monitor:
            print(f"Final ssr (weighted): {reg_info['ssr']}")

        reg_info['total_points'] = total_points
        reg_info['total_params'] = (self.num_var_ks
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

            if monitor:
                print("Bootstrapping simulations:")
                boot_CI_results = Parallel(n_jobs=boot_nodes)(
                        delayed(_sim_boot)(d)
                                for d in tqdm(list(range(num_datasets))))
            else:
                boot_CI_results = Parallel(n_jobs=boot_nodes)(
                        delayed(_sim_boot)(d)
                                for d in list(range(num_datasets)))

            reg_info['boot_CI'] = boot_CI
            reg_info['boot_param_CIs'] = []
            reg_info['boot_plot_CIs'] = []
            reg_info['boot_plot_ts'] = []
            reg_info['boot_calc_CIs'] = []
            for b in boot_CI_results:
                reg_info['boot_param_CIs'].append(b[1])
                reg_info['boot_plot_CIs'].append(b[2])
                reg_info['boot_calc_CIs'].append(b[3])
                reg_info['boot_plot_ts'].append(b[4])

            if cc_ints:
                reg_info['conf_contours'] = self.confidence_contours(
                        reg_info, datasets, num_datasets, cc_ints,
                        cc_mult=cc_mult, monitor=monitor,
                        nodes=boot_nodes, include_cs=cc_include_cs)
        return reg_info

    def confidence_contours(self, reg_info, datasets, num_datasets,
                            num_intervals, cc_mult=2.0, monitor=False,
                            nodes=-1, include_cs=False):
        """Generates confidence contour data around each pair of fit
        parameters.

        Returns a tuple of the two indices followed by the data as a list
        of tuples (p1, p2, ssr).
        """

        def _results(inp):
            """Reformats _residual_fix for parallel processing.
            """
            (p1, p2), p1_ind, p2_ind, var_params, var_params_ind, bounds_del = inp # change line
            cc_results = scipy.optimize.least_squares(
                    self._residual_fix, var_params,
                    bounds=bounds_del, # change argument
                    args=(var_params_ind, [p1, p2],
                          const_params_ind, datasets,
                          reg_info['parameter_constants'], False))
            ssr = cc_results.cost * 2
            return p1, p2, ssr

        ks_bot = list(reg_info['boot_param_CIs'][0][0][0])
        ks_top = list(reg_info['boot_param_CIs'][0][0][1])
        if include_cs:
            all_conc0_names = ([f"{n}({i+1})" for i in range(num_datasets)
                                for n in self.conc0_var_names])
            all_parameter_names = self.k_var_names + all_conc0_names

            # Flattens list of lists of cs.
            cs_bot = list(itertools.chain.from_iterable([list(
                    reg_info['boot_param_CIs'][d][1][0])
                    for d in range(num_datasets)]))
            cs_top = list(itertools.chain.from_iterable([list(
                    reg_info['boot_param_CIs'][d][1][1])
                    for d in range(num_datasets)]))
            all_params_bot = ks_bot + cs_bot
            all_params_top = ks_top + cs_top
            total_num_params = (self.num_var_ks
                                + self.num_var_concs0*num_datasets)
        else:  # Only ks are included.
            all_parameter_names = self.k_var_names
            all_params_bot = ks_bot
            all_params_top = ks_top
            total_num_params = self.num_var_ks

        results = []
        for p1_ind in range(total_num_params-1):
            for p2_ind in range(p1_ind+1, total_num_params):
                p1_vals = self._bracket_param(
                        reg_info['all_params'][p1_ind],
                        all_params_bot[p1_ind],
                        all_params_top[p1_ind],
                        num_intervals,
                        p1_ind, # add argument
                        cc_mult=cc_mult)
                p2_vals = self._bracket_param(
                        reg_info['all_params'][p2_ind],
                        all_params_bot[p2_ind],
                        all_params_top[p2_ind],
                        num_intervals,
                        p2_ind, # add argument
                        cc_mult=cc_mult)

                var_params = list(reg_info['all_params'])
                var_params_ind = [x for x in range(len(
                        reg_info['all_params']))]
                for r in sorted([p1_ind, p2_ind], reverse=True):
                    del var_params[r]
                    del var_params_ind[r]
                const_params_ind = [p1_ind, p2_ind]

                # add lines
                bounds_del = copy.deepcopy(self.bounds) 
                if self.get_elements_of_nested_list(self.bounds) > 1: # bounds also should be deleted when bounds are specified
                    for r in sorted([p1_ind, p2_ind], reverse=True):
                        del bounds_del[0][r]
                        del bounds_del[1][r]

                p_combos = [p for p in itertools.product(p1_vals, p2_vals)]
                total_p_combos = len(p_combos)

                if monitor:
                    print(f"Confidence contours for "
                          f"{all_parameter_names[p1_ind]} and "
                          f"{all_parameter_names[p2_ind]}:")
                    cc_results = Parallel(n_jobs=nodes)(
                            delayed(_results)((ps, p1_ind, p2_ind, var_params,
                                           var_params_ind, bounds_del)) for ps in tqdm(p_combos)) # change argument
                    results.append([(all_parameter_names[p1_ind],
                                     all_parameter_names[p2_ind]),
                                     cc_results])
                else:
                    cc_results = Parallel(n_jobs=nodes)(
                            delayed(_results)((ps, p1_ind, p2_ind, var_params,
                                           var_params_ind,bounds_del)) for ps in p_combos) # change argument
                    results.append([(all_parameter_names[p1_ind],
                                     all_parameter_names[p2_ind]),
                                     cc_results])

        return results

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
                  monitor=False, nodes=-1):
        """Process a set of datasets obtained by a bootstrapping method,
        returning the ks and concs.

        """
        def _results(inp):
            """Used to parallelize fitting of datasets.

            """
            datasets = inp
            fit = scipy.optimize.least_squares(
                    self._residual, fit_params, bounds=self.bounds,
                    args=(datasets, constants, False))['x']
            return fit

        total_datasets = len(all_datasets)
        num_datasets = len(all_datasets[0])

        if monitor:
            print("Bootstrapping fits:")
            boot_params = Parallel(n_jobs=nodes)(
                    delayed(_results)(d) for d in tqdm(all_datasets))
        else:
            boot_params = Parallel(n_jobs=nodes)(
                    delayed(_results)(d) for d in all_datasets)

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
        smooth_ts_out = reg_info['predicted_time'][dataset_n]

        boot_iterations = reg_info['boot_num']
        CI_num = math.ceil(boot_iterations*(100-CI)/200)
        assert CI_num > 0

        plot_topCI = np.empty((0, num_points, self.num_data_concs))
        plot_botCI = np.empty((0, num_points, self.num_data_concs))
        calc_topCI = [np.empty(0) for _ in range(self.num_calcs)]
        calc_botCI = [np.empty(0) for _ in range(self.num_calcs)]
        for n in range(boot_iterations):
            if monitor:
                print(f"Bootstrapping simulation {n+1} "
                      f"of {reg_info['boot_num']}", end="")
            _, boot_plot, _, boot_calcs = self.simulate(
                    list(reg_info['boot_fit_ks'][n]) + reg_info['fixed_ks'],
                    (list(reg_info['boot_fit_concs'][dataset_n][n])
                     + reg_info['fixed_concs'][dataset_n]),
                    smooth_ts_out, integrate=True, calcs=True) # chagne line
            if (n+1) <= CI_num:
                plot_topCI = np.append(plot_topCI, [boot_plot], axis=0)
                plot_topCI = np.sort(plot_topCI, axis=0)
                plot_botCI = np.append(plot_botCI, [boot_plot], axis=0)
                plot_botCI = np.sort(plot_botCI, axis=0)
                for i in range(self.num_calcs):
                    calc_topCI[i] = np.append(calc_topCI[i], boot_calcs[i][1])
                    calc_topCI[i] = np.sort(calc_topCI[i])
                    calc_botCI[i] = np.append(calc_botCI[i], boot_calcs[i][1])
                    calc_botCI[i] = np.sort(calc_botCI[i])
            else:
                if not np.all(
                        np.maximum(plot_topCI[0], boot_plot) == plot_topCI[0]):
                    if monitor:
                        print(', top CI increased', end="")
                    plot_topCI = np.append(plot_topCI, [boot_plot], axis=0)
                    plot_topCI = np.sort(plot_topCI, axis=0)
                    plot_topCI = plot_topCI[1:]
                if not np.all(
                        np.minimum(plot_botCI[-1], boot_plot) == plot_botCI[-1]):
                    if monitor:
                        print(', bottom CI decreased', end="")
                    plot_botCI = np.append(plot_botCI, [boot_plot], axis=0)
                    plot_botCI = np.sort(plot_botCI, axis=0)
                    plot_botCI = plot_botCI[:-1]
                for i in range(self.num_calcs):
                    if boot_calcs[i][1] > calc_topCI[i][0]:
                        calc_topCI[i] = np.append(calc_topCI[i], boot_calcs[i][1])
                        calc_topCI[i] = np.sort(calc_topCI[i])
                        calc_topCI[i] = calc_topCI[i][1:]
                    if boot_calcs[i][1] < calc_botCI[i][-1]:
                        calc_botCI[i] = np.append(calc_botCI[i], boot_calcs[i][1])
                        calc_botCI[i] = np.sort(calc_botCI[i])
                        calc_botCI[i] = calc_botCI[i][:-1]
            if monitor:
                print()

        calc_top_cutoffs = [c[0] for c in calc_topCI]
        calc_bot_cutoffs = [c[0] for c in calc_botCI]
        return ((plot_topCI[0], plot_botCI[-1]),
                (calc_top_cutoffs, calc_bot_cutoffs),
                smooth_ts_out)

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

    def _residual_fix(self, parameters, parameters_ind, fixed_parameters,
                      fixed_parameters_ind, datasets, constants,
                      monitor=False):
        """Modified version of residual that allows parameters to be fixed.

        parameters: list of parameters that will be optimized.
        parameters_ind: list of indices for the parameters that will be
            optimized.
        fixed_parameters: list of parameters that will not be optimized.
        fixed_parameters_ind: list of indices for the parameters that will not
            be optimized.
        """

        rebuilt_params = [0]*(len(parameters_ind) + len(fixed_parameters_ind))
        for n in range(len(parameters_ind)):
            rebuilt_params[parameters_ind[n]] = parameters[n]
        for n in range(len(fixed_parameters_ind)):
            rebuilt_params[fixed_parameters_ind[n]] = fixed_parameters[n]

        return self._residual(rebuilt_params, datasets, constants, monitor)

    @staticmethod
    def _dataset_concs(n, c_list, c_per_n):
        """Returns the concentrations corresponding to a specific
        dataset n from a list of concs for all datasets.

        """

        return c_list[(n*c_per_n):(n*c_per_n + c_per_n)]

    @staticmethod
    def get_all_models(model_search_dirs):
        """Returns a dictionary of all available models.
        """
        direct_models_params = {}
        indirect_models_params = {}

        for directory in model_search_dirs:
            # Check if directory exists.
            if os.path.isdir(directory):
                for filename in os.listdir(directory):
                    path_to_file = os.path.join(directory, filename)
                    if (os.path.isfile(path_to_file)
                            and filename.endswith(constants.MODEL_FILE_EXT)):
                        with open(path_to_file) as file:
                            model_params = yaml.load(file.read(),
                                                     Loader=yaml.FullLoader)
                        if model_params.get('type') == "indirect":
                            if model_params['name'] not in indirect_models_params:
                                indirect_models_params[model_params['name']] = model_params
                                indirect_models_params[model_params['name']]['path'] = path_to_file
                        else:
                            if model_params['name'] not in direct_models_params:
                                direct_models_params[model_params['name']] = model_params
                                direct_models_params[model_params['name']]['path'] = path_to_file

        all_models = {}

        for m in direct_models_params:
            new_model = KineticModel(**direct_models_params[m])
            all_models[new_model.name] = new_model

        for m in indirect_models_params:
            parent_model_name = indirect_models_params[m]['parent_model_name']
            parent_model = all_models[parent_model_name]
            new_model = IndirectKineticModel(
                    parent_model=parent_model, **indirect_models_params[m])
            all_models[new_model.name] = new_model

        return all_models

    @staticmethod
    def get_model(model_name, all_models):
        """Returns the model object corresponding to model_name.
        """
        try:
            return all_models[model_name]
        except KeyError:
            print(f'"{model_name}" is not a valid model.')
            print(", ".join(a for a in all_models), "are currently available.")
            sys.exit(1)

    def _bracket_param(self, param, low, high, num_iterations, ind, cc_mult=2):
        """Used to generated confidence contours. Returns the upper and
        lower limits that should be used for a given value of param and
        high and low CIs.
        """
        delta_high = high - param
        delta_low = param - low
        delta = cc_mult*max(delta_high, delta_low)

        # We want the CCs to always include the actual optimized param,
        # ideally in the center but offset if the parameter would be
        # outside of the bounds. If an even number of iterations is
        # specified, such that there is no exact center of the plot, the
        # range favors the high end.

        if self.get_elements_of_nested_list(self.bounds) == 1:
            ub = self.bounds[1]
            lb = self.bounds[0]
        else:
            ub = self.bounds[1][ind]
            lb = self.bounds[0][ind]

        if ((param + delta) <= ub and
                (param - delta) >= lb): # ideal case
            if num_iterations % 2 == 1:
                interval = 2*delta/(num_iterations - 1)
                bottom = param - (num_iterations-1)/2 * interval
            else:
                interval = 2*delta/num_iterations
                bottom = param - (num_iterations/2 - 1) * interval
        elif ((lb + 2*delta) <= ub and
                (param - delta) < lb):
            # Bounded on low end.
            max_interval = 2*delta/(num_iterations - 1)
            # Integer number of intervals to the actual param value.
            n_to_p = math.ceil((param - lb)/max_interval)
            interval = (param - lb)/n_to_p
            bottom = lb
        elif ((param + delta) > ub and
                (ub - 2*delta) >= lb):
            # Bounded on high end.
            max_interval = 2*delta/(num_iterations - 1)
            n_to_p = math.ceil((ub - param)/max_interval)
            interval = (ub - param)/n_to_p
            bottom = ub - (num_iterations-1)*interval
        else:
            # Bounded on both sides.
            max_interval = (ub - lb)/(num_iterations - 1)
            n_to_p_low = math.ceil((param - lb)/max_interval) # number of interval in lb ~ param
            n_to_p_high = math.ceil((ub - param)/max_interval) # number of interval in param ~ ub
            if n_to_p_low < n_to_p_high: # biased to lb
                n_to_p = math.ceil(n_to_p_low)
                interval = (param - lb)/n_to_p
                bottom = lb
            else: # biased to ub
                n_to_p = math.ceil(n_to_p_high)
                interval = (ub - param)/n_to_p
                bottom = param - n_to_p*interval

        assert ((bottom >= lb)
                and (bottom + interval*(num_iterations - 1)) <= ub)

        return [bottom + i*interval for i in range(num_iterations)]


class IndirectKineticModel(KineticModel):
    """Defines an indirect kinetic model: one in which the observed data
    does not map cleanly onto species' concentrations.

    """

    def __init__(self,
                 name,
                 parent_model,
                 description,
                 species,
                 parent_model_name=None,
                 type="indirect",
                 int_eqn=[],
                 int_eqn_desc=[],
                 calcs=[],
                 calcs_desc=[],
                 lifetime_conc=[],
                 lifetime_fracs=[1, 1/2.71828, 0.1, 0.01],
                 rectime_conc=[],
                 rectime_fracs=[0.99],
                 path=None,
                 **kwargs
                 ):

        self.parent_model = parent_model

        conc_mapping = [s['map'] for s in species]
        map_lambda = CONC_MAP_LAMBDA[0] + ",".join(conc_mapping) + CONC_MAP_LAMBDA[1]
        self.conc_mapping = eval(map_lambda)

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

        self.legend_names = []
        self.sort_order = []
        self.top_plot = []
        self.bottom_plot = []
        for n in range(len(species)):
            self.legend_names.append(species[n]['name'])
            if 'sort' in species[n]:
                self.sort_order.append(species[n]['sort'])
            if species[n]['plot'] == "top":
                self.top_plot.append(n)
            elif species[n]['plot'] == "bottom":
                self.bottom_plot.append(n)

        self.int_eqn = self.parent_model.int_eqn
        self.int_eqn_desc = self.parent_model.int_eqn_desc
        self.calcs = self.parent_model.calcs
        self.calcs_desc = self.parent_model.calcs_desc
        self.weight_func = self.parent_model.weight_func
        self.bounds = self.parent_model.bounds
        self.lifetime_conc = self.parent_model.lifetime_conc
        self.lifetime_fracs = self.parent_model.lifetime_fracs
        self.rectime_conc = self.parent_model.rectime_conc
        self.rectime_fracs = self.parent_model.rectime_fracs
        self.path = path

    @property
    def num_data_concs(self):
        return len(self.legend_names)

    def _solved_kin_sys(self, conc0, ks, times):
        parent_model_soln = scipy.integrate.odeint(
                self.kin_sys, conc0, times, args=tuple(ks),
                mxstep=self.MAX_STEPS)

        return self.conc_mapping(parent_model_soln.transpose())

    def simulate(self, ks, concs, smooth_ts_out, integrate=False,
                 calcs=False):
        """Overriding the simulate function in this way allows calculations
        from the parent model to be executed.
        """
        smooth_ts_out, smooth_curves_out, integrals, calc_results = (
                self.parent_model.simulate(
                        ks, concs, smooth_ts_out, integrate, calcs))

        return (smooth_ts_out,
                self.conc_mapping(smooth_curves_out.transpose()),
                integrals,
                calc_results)
