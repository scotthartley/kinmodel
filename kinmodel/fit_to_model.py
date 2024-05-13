"""Interfaces with the KineticModel class to fit experimental data to a
given kinetic model and output the results.

** Modified by Gyunam Park 24.02.28

"""
import platform
import numpy as np
import scipy
import pickle
from matplotlib import pyplot as plt, rcParams
from . import _version
from .Dataset import Dataset
from .simulate_model import simulate_and_output
from .KineticModel import IndirectKineticModel

# Parameters and settings for plots.
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
MARKER_SIZE = 12
FIGURE_SIZE_1 = (3.3, 3)
FIGURE_SIZE_2 = (3.3, 5.2)
YLABEL = "C"
XLABEL = "t"
PARAM_LOC = [0.65, 0.1]
# Original CONTOUR_LEVELS; did not provide sufficient discrimination in
# areas of interest.
# CONTOUR_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05]
CONTOUR_LEVELS = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68,
                  0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88,
                  0.9, 0.92, 0.94, 0.96, 0.98, 1.00]
CONTOUR_TICKS = [0, 0.25, 0.5, 0.75, 1.0]
TICK_ALIGN_THRESHOLD = 10

PLT_PARAMS = {'font.size': 8,
              'font.family': 'sans-serif',
              'font.sans-serif': ['Arial'],
              'lines.linewidth': 0.75,
              'axes.linewidth': 0.75,
              'legend.frameon': False,
              'legend.fontsize': 6,
              'image.cmap': 'RdGy_r',  # "plasma" also nice
              'image.origin': 'lower',
              'image.interpolation': 'nearest',
              }

PICKLE_SUFFIX = "_reg_info.pickle"

# Prevent line breaking and format numbers from np
np.set_printoptions(linewidth=np.nan)
np.set_printoptions(precision=2, suppress=True)


def prepare_text(    
        model, reg_info, ds_num, num_points, time_exp_factor, filename="",
        full_simulation=True, more_stats=False):
    """Generates the output text.

    The number of points (num_points) for the output simulation and
    integrals must be specified.

    """
    smooth_ts_out, smooth_curves_out, integrals, calculations = model.simulate(
            reg_info['fit_ks'] + reg_info['fixed_ks'],
            reg_info['fit_concs'][ds_num] + reg_info['fixed_concs'][ds_num],
            reg_info['predicted_time'][ds_num], # change argument
            integrate=True, calcs=True)

    num_ks = len(reg_info['fit_ks'])
    num_concs = len(reg_info['fit_concs'][ds_num])
    # Starting index for dataset-specific concentration parameters.
    conc0_i = num_ks + num_concs*ds_num

    dataset_params = reg_info['fit_ks'] + reg_info['fit_concs'][ds_num]
    dataset_consts = reg_info['fixed_ks'] + reg_info['fixed_concs'][ds_num]

    cov_stddevs = (
            reg_info['cov_errors'][:num_ks].tolist()
            + reg_info['cov_errors'][conc0_i:conc0_i+num_concs].tolist())
    if 'boot_num' in reg_info:
        boot_k_CIs, boot_conc_CIs = reg_info['boot_param_CIs'][ds_num]
        param_CIs = np.append(boot_k_CIs, boot_conc_CIs, axis=1)

    # List of all parameter names, for labeling matrices with all fit
    # parameters included (not just the ones specific to this dataset).
    all_par_names = model.parameter_names[:num_ks]
    all_par_names += ['']*num_concs*ds_num
    all_par_names += model.parameter_names[num_ks:]
    all_par_names += ['']*num_concs*(reg_info['num_datasets']-ds_num-1)

    if reg_info['dataset_names'][ds_num]:
        title = ("Regression results for dataset "
                 f"{reg_info['dataset_names'][ds_num]} "
                 f"from file \"{filename}\"")
    else:
        title = f"Regression results for file \"{filename}\""

    text = title + "\n"
    text += "="*len(title) + "\n"

    text += f"Python version: {platform.python_version()}\n"
    text += f"Numpy version: {np.version.version}\n"
    text += f"Scipy version: {scipy.version.version}\n"
    text += f"kinmodel version: {_version.__version__}\n"
    text += "\n"
    text += "\n"

    text += "Model\n"
    text += "-----\n"
    text += f"Name: {model.name}\n"
    text += f"Description: {model.description}\n"
    text += "\n"
    text += "\n"

    text += "Regression\n"
    text += "----------\n"

    text += "Optimized parameters:\n"
    text += "\n"
    if 'boot_num' in reg_info:
        for n in range(len(dataset_params)):
            text += (f"{model.parameter_names[n]:>{model.len_params}} "
                     f"= {dataset_params[n]:+5e} "
                     f"± {(param_CIs[1][n]-param_CIs[0][n])/2:.1e} "
                     f"({param_CIs[0][n]:+5e}, {param_CIs[1][n]:+5e})\n")
        text += "\n"
        text += (f"Errors are {reg_info['boot_CI']}% confidence intervals from "
                 f"bootstrapping using the {reg_info['boot_method']} method "
                 f"({reg_info['boot_num']} permutations).\n")
        if reg_info['boot_force1st']:
            text += (f"Initial points retained in all bootstrap "
                     f"permutations.\n")
    else:
        for n in range(model.num_params):
            text += (f"{model.parameter_names[n]:>{model.len_params}} "
                     f"= {dataset_params[n]:+5e}\n")
    text += "\n"

    text += "Fixed parameters:\n"
    text += "\n"
    for n in range(model.num_consts):
        text += (f"{model.constant_names[n]:>{model.len_consts}} "
                 f"= {dataset_consts[n]:+5e}\n")
    text += "\n"

    text += "Regression info:\n"
    text += "\n"
    text += f"Success: {reg_info['success']}\n"
    text += f"Message: {reg_info['message']}\n"

    text += (f"Total points (dof): {reg_info['total_points']} "
             f"({reg_info['dof']})\n")
    text += f"Sum square residuals (weighted): {reg_info['ssr']:.4e}\n"
    text += f"Sum square residuals (unweighted): {reg_info['pure_ssr']:.4e}\n"
    text += f"RMSD (unweighted): {reg_info['pure_rmsd']:.2e}\n"
    text += "\n"

    if more_stats:
        text += "Covariance matrix:\n"
        for n in range(reg_info['total_params']):
            text += (f"{all_par_names[n]:>{model.len_params}} "
                     + " ".join(f"{m:+.2e}" for m in reg_info['pcov'][n])
                     + "\n")
        text += "\n"

        text += "Parameter σ from covariance matrix (√diagonal):\n"
        text += " ".join(f"{m:+.1e}" for m in cov_stddevs) + "\n"
        text += "\n"

        text += "Correlation matrix:\n"
        for n in range(reg_info['total_params']):
            text += (f"{all_par_names[n]:>{model.len_params}} "
                     + " ".join(f"{m:+.2f}" for m in reg_info['corr'][n])
                     + "\n")
        text += "\n"
    text += "\n"

    text += "Simulation\n"
    text += "----------\n"
    text += (f"Points used to generate integrals and concentration vs time "
             f"data: {num_points}\n")
    text += "\n"

    if integrals:
        text += "Integrals:\n"
        text += "\n"
        for integral in integrals:
            integral_label = "∫ " + integral[0] + " dt"
            text += (f"{integral_label:>{model.len_int_eqn_desc+5}} "
                     f"= {integral[1]:+5e}\n")
        text += "\n"

    if calculations:
        text += "Calculations:\n"
        text += "\n"
        if 'boot_num' in reg_info:
            for n in range(len(calculations)):
                calc_label = calculations[n][0]
                text += (f"{calc_label:>{model.len_calcs_desc}} "
                         f"= {calculations[n][1]:+5e} "
                         f"({reg_info['boot_calc_CIs'][ds_num][1][n]:+5e}, "
                         f"{reg_info['boot_calc_CIs'][ds_num][0][n]:+5e})\n")
        else:
            for calc in calculations:
                calc_label = calc[0]
                text += (f"{calc_label:>{model.len_calcs_desc}} "
                         f"= {calc[1]:+5e}\n")
        text += "\n"

    if full_simulation:
        text += "Results:\n"
        text += "\n"
        if 'boot_num' in reg_info:
            assert (list(reg_info['boot_plot_ts'][ds_num])
                    == list(smooth_ts_out)), ("Simulation and bootstrap"
                                              "points do not line up!")
            boot_CI_data = reg_info['boot_plot_CIs'][ds_num]
            text += ("t " + " ".join(model.legend_names) + " "
                     + " ".join(f"{n}CI+ {n}CI−" for n in model.legend_names)
                     + "\n")
            for n in range(len(smooth_ts_out)):
                best_fit_points = " ".join(
                        str(m) for m in smooth_curves_out[n])
                CI_points = " ".join(
                        [f"{str(pCI)} {str(mCI)}" for pCI, mCI in
                         zip(boot_CI_data[0][n], boot_CI_data[1][n])])
                text += (str(smooth_ts_out[n]) + " " + best_fit_points + " "
                         + CI_points + "\n")
        else:
            text += "t " + " ".join(model.legend_names) + "\n"

            for n in range(len(smooth_ts_out)):
                text += str(smooth_ts_out[n]) + " " + " ".join(
                    str(m) for m in smooth_curves_out[n]) + "\n"

    return text


def generate_plot(model, reg_info, ds_num, num_points, time_exp_factor,
                  output_filename, boot_CI=95, common_y=True, no_params=False,
                  units=None, plot_semilogx=True):
    """Generates the output plot.

    Number of points must be specified. Saved as pdf to output_filename.

    """
    rcParams.update(PLT_PARAMS)

    dataset_params = reg_info['fit_ks'] + reg_info['fit_concs'][ds_num]
    dataset_consts = reg_info['fixed_ks'] + reg_info['fixed_concs'][ds_num]

    # delete and change argument
    smooth_ts_plot, smooth_curves_plot, _, _ = model.simulate(
            reg_info['fit_ks'] + reg_info['fixed_ks'],
            reg_info['fit_concs'][ds_num] + reg_info['fixed_concs'][ds_num],
            reg_info['predicted_time'][ds_num], integrate=False, calcs=False)

    if 'boot_num' in reg_info and boot_CI:
        boot_CI_plots = reg_info['boot_plot_CIs'][ds_num]
        boot_ts = reg_info['boot_plot_ts'][ds_num]

    if model.top_plot:
        plt.figure(figsize=FIGURE_SIZE_2)
    else:
        plt.figure(figsize=FIGURE_SIZE_1)

    if model.top_plot:
        plt.subplot(211)
        col = 0
        for n in ([np.array(reg_info['dataset_concs'][ds_num]).T[m]
                   for m in model.top_plot]):
            plt.scatter(
                    reg_info['dataset_times'][ds_num], n, c=COLORS[col],
                    s=MARKER_SIZE, linewidths=0)
            col += 1

        col = 0
        for n in [smooth_curves_plot.T[m] for m in model.top_plot]:
            plt.plot(smooth_ts_plot, n, COLORS[col] + '-')
            col += 1

        if 'boot_num' in reg_info and boot_CI:
            col = 0
            for n in [boot_CI_plots[0].T[m] for m in model.top_plot]:
                plt.plot(boot_ts, n, COLORS[col] + ':')
                col += 1
            col = 0
            for n in [boot_CI_plots[1].T[m] for m in model.top_plot]:
                plt.plot(boot_ts, n, COLORS[col] + ':')
                col += 1

        plt.legend([model.legend_names[n] for n in model.top_plot], loc=4)

        if common_y:
            _, y_buffer = plt.margins()
            ymax = max(
                    max(reg_info['max_exp_concs'][n] for n in model.top_plot),
                    max(reg_info['max_pred_concs'][n] for n in model.top_plot)
                    )*(1 + y_buffer)
            plt.ylim(ymin=0, ymax=ymax)
        else:
            plt.ylim(ymin=0)
        plt.xlim(xmin=0, xmax=smooth_ts_plot[-1])

        if units:
            plt.ylabel(f"{YLABEL} ({units[1]})")
        else:
            plt.ylabel(YLABEL)     

        if plot_semilogx:
            plt.xscale("log")
            if smooth_ts_plot[0]==0:
                plt.xlim(xmin=smooth_ts_plot[1], xmax=smooth_ts_plot[-1])
            else:
                plt.xlim(xmin=smooth_ts_plot[0], xmax=smooth_ts_plot[-1])

    if model.bottom_plot:
        if model.top_plot:
            plt.subplot(212)
        else:
            plt.subplot(111)

        col = 0
        for n in ([np.array(reg_info['dataset_concs'][ds_num]).T[m]
                   for m in model.bottom_plot]):
            plt.scatter(reg_info['dataset_times'][ds_num], n, c=COLORS[col],
                        s=MARKER_SIZE, linewidths=0, zorder=2)
            col += 1

        col = 0
        for n in [smooth_curves_plot.T[n] for n in model.bottom_plot]:
            plt.plot(smooth_ts_plot, n, COLORS[col] + '-', zorder=3)
            col += 1

        if 'boot_num' in reg_info and boot_CI:
            col = 0
            for n in [boot_CI_plots[0].T[m] for m in model.bottom_plot]:
                plt.plot(boot_ts, n, COLORS[col] + ':')
                col += 1
            col = 0
            for n in [boot_CI_plots[1].T[m] for m in model.bottom_plot]:
                plt.plot(boot_ts, n, COLORS[col] + ':')
                col += 1

        plt.legend([model.legend_names[n] for n in model.bottom_plot], loc=2)

        if common_y:
            _, y_buffer = plt.margins()
            ymax = max(
                    max(reg_info['max_exp_concs'][n] for n in model.bottom_plot),
                    max(reg_info['max_pred_concs'][n] for n in model.bottom_plot)
                    )*(1 + y_buffer)
            plt.ylim(ymin=0, ymax=ymax)
        else:
            plt.ylim(ymin=0)
        plt.xlim(xmin=0, xmax=smooth_ts_plot[-1])

        if units:
            plt.xlabel(f"{XLABEL} ({units[0]})")
            plt.ylabel(f"{YLABEL} ({units[1]})")
        else:
            plt.xlabel(XLABEL)
            plt.ylabel(YLABEL)
        
        if plot_semilogx:
            plt.xscale("log")
            if smooth_ts_plot[0]==0:
                plt.xlim(xmin=smooth_ts_plot[1], xmax=smooth_ts_plot[-1])
            else:
                plt.xlim(xmin=smooth_ts_plot[0], xmax=smooth_ts_plot[-1])

        # Print parameters on plot.
        if not no_params:
            pars_to_print = ""
            for n in range(model.num_params):
                pars_to_print += "{} = {:.2e}\n".format(
                        model.parameter_names[n], dataset_params[n])
            for n in range(model.num_consts):
                pars_to_print += "{} = {:.2e}\n".format(
                        model.constant_names[n], dataset_consts[n])

            plt.text(PARAM_LOC[0], PARAM_LOC[1], pars_to_print,
                     transform=plt.gca().transAxes,
                     fontsize=rcParams['legend.fontsize'])

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def prepare_conf_contours(pair):
    """Generates the output text for confidence contours.
    """
    text = f"{pair[0][0]} {pair[0][1]} ssr\n"
    for result in pair[1]:
        text += " ".join(str(t) for t in result)
        text += "\n"
    return text


def generate_cc_plot(pair, num_points, reg_info, output_base_filename,
                     output_contour_plot=False):
    """Generates contour plots for confidence contours.
    """
    rcParams.update(PLT_PARAMS)
    data = np.array(pair[1])
    xlist = data[:, 0]
    ylist = data[:, 1]
    zlist = data[:, 2]
    # Plotted data is actually the normalized inverse of the ssr.
    zlist_inv = reg_info['ssr']/zlist
    X = [xlist[n] for n in range(0, len(xlist), num_points)]
    Y = ylist[:num_points]
    Z = np.reshape(zlist_inv, (num_points, num_points)).T

    # Generate contour plot
    if output_contour_plot:
        plt.figure(figsize=FIGURE_SIZE_1)
        # cp = plt.contour(X, Y, Z, CONTOUR_LEVELS, colors='black')
        cpf = plt.contourf(X, Y, Z, CONTOUR_LEVELS)
        plt.colorbar(cpf, ticks=CONTOUR_TICKS)
        plt.xlabel(pair[0][0])
        plt.ylabel(pair[0][1])
        plt.tight_layout()
        plt.savefig(output_base_filename + "_c.pdf")
        plt.close()

    # Generate heatmap
    plt.figure(figsize=FIGURE_SIZE_1)
    # hm = plt.imshow(Z, extent=[X[0], X[-1], Y[0], Y[-1]], aspect='auto',
    #                 vmax=CONTOUR_LEVELS[-1], vmin=CONTOUR_LEVELS[0])
    hm = plt.imshow(Z, aspect='auto',
                    vmax=CONTOUR_LEVELS[-1], vmin=CONTOUR_LEVELS[0])
    ax = plt.gca()
    ax.set_xticks([0, len(X)-1])
    ax.set_xticklabels([f"{X[0]:.1e}", f"{X[-1]:.1e}"])
    ax.set_yticks([0, len(Y)-1])
    ax.set_yticklabels([f"{Y[0]:.1e}", f"{Y[-1]:.1e}"])
    plt.colorbar(hm, ticks=CONTOUR_TICKS)
    if num_points > TICK_ALIGN_THRESHOLD:
        plt.setp(ax.get_yticklabels()[0], rotation=90, ha="left", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels()[-1], rotation=90, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_xticklabels()[0], ha="left", rotation_mode="anchor")
        plt.setp(ax.get_xticklabels()[-1], ha="right", rotation_mode="anchor")
    else:
        plt.setp(ax.get_yticklabels()[0], rotation=90, ha="center", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels()[-1], rotation=90, ha="center", rotation_mode="anchor")
    plt.xlabel(pair[0][0])
    plt.ylabel(pair[0][1])
    plt.tight_layout()
    plt.savefig(output_base_filename + '_hm.pdf')
    plt.close()


def fit_and_output(
            model,
            data_filename,
            fixed_ks=None,
            fixed_concs=None,
            k_guesses=None,
            conc_guesses=None,
            text_output_points=3000,
            text_time_extension_factor=3.0,
            text_output=True,
            plot_output_points=1000,
            plot_time_extension_factor=1.1,
            text_full_output=True,
            monitor=False,
            bootstrap_iterations=100,
            bootstrap_CI=95,
            bootstrap_force1st=False,
            bootstrap_nodes=None,
            confidence_contour_intervals=None,
            confidence_contour_multiplier=3.0,
            confidence_contour_cs=False,
            confidence_contour_include_ccplot=False,
            more_stats=False,
            common_y=True,
            plot_no_params=False,
            units=None,
            simulate=True,
            calcs=True,
            load_reg_info=False,
            plot_semilogx=True): # New code
    """Carry out the fit of the model and output the data.

    """

    if not load_reg_info:
        datasets = Dataset.read_raw_data(model, data_filename)
        reg_info = model.fit_to_model(
                datasets,
                ks_guesses=k_guesses,
                conc0_guesses=conc_guesses,
                ks_const=fixed_ks,
                conc0_const=fixed_concs,
                monitor=monitor,
                N_boot=bootstrap_iterations,
                boot_CI=bootstrap_CI,
                boot_points=text_output_points,
                boot_t_exp=text_time_extension_factor,
                boot_force1st=bootstrap_force1st,
                boot_nodes=bootstrap_nodes,
                cc_ints=confidence_contour_intervals,
                cc_mult=confidence_contour_multiplier,
                cc_include_cs=confidence_contour_cs,
                plot_semilogx=plot_semilogx) # New code

        file_suffix = ""
        if bootstrap_force1st:
            file_suffix += "_ff"

        base_filename = f"{data_filename}_{model.name}{file_suffix}"

        with open(base_filename + PICKLE_SUFFIX, 'wb') as file:
            pickle.dump(reg_info, file)
    else:
        with open(data_filename, 'rb') as file:
            reg_info = pickle.load(file)

        if confidence_contour_intervals:
            reg_info['conf_contours'] = model.confidence_contours(
                    reg_info, reg_info['datasets'],
                    reg_info['num_datasets'],
                    confidence_contour_intervals,
                    cc_mult=confidence_contour_multiplier, monitor=monitor,
                    nodes=bootstrap_nodes, include_cs=confidence_contour_cs)

        base_filename = f"{data_filename}"

    for n in range(reg_info['num_datasets']):
        output_text = prepare_text(
                model, reg_info, n, text_output_points,
                text_time_extension_factor, data_filename,
                text_full_output, more_stats)
        if text_output:
            text_filename = (f"{base_filename}"
                             f"_{reg_info['dataset_names'][n]}.txt")
            with open(text_filename, 'w', encoding='utf-8') as write_file:
                print(output_text, file=write_file)
        else:
            print(output_text)

    if plot_output_points:
        for n in range(reg_info['num_datasets']):
            plot_filename = (f"{base_filename}"
                             f"_{reg_info['dataset_names'][n]}.pdf")
            generate_plot(model, reg_info, n, plot_output_points,
                          plot_time_extension_factor, plot_filename,
                          bootstrap_CI, common_y, plot_no_params, units, plot_semilogx) # add argument

    if (type(model) is IndirectKineticModel) and simulate:
        for n in range(reg_info['num_datasets']):
            sim_filename = (f"{base_filename}_sim_"
                            f"{reg_info['dataset_names'][n]}")
            simulate_and_output(
                    model=model.parent_model,
                    ks=reg_info['fit_ks'] + reg_info['fixed_ks'],
                    concs=(reg_info['fit_concs'][n]
                           + reg_info['fixed_concs'][n]),
                    time=reg_info['predicted_time'][n], # change input variable
                    text_num_points=text_output_points,
                    plot_num_points=plot_output_points,
                    filename=sim_filename,
                    text_full_output=True,
                    units=units,
                    plot_time=reg_info['predicted_time'][n]) # change input variable

    if confidence_contour_intervals:
        base_cc_filename = base_filename + "_cc"
        for param_pair in reg_info['conf_contours']:
            cc_filename = (base_cc_filename +
                           f"_{param_pair[0][0]}-{param_pair[0][1]}")
            cc_output_text = prepare_conf_contours(param_pair)
            cc_text_filename = cc_filename + ".txt"
            with open(cc_text_filename, 'w', encoding='utf-8') as write_file:
                print(cc_output_text, file=write_file)

            generate_cc_plot(
                    param_pair, confidence_contour_intervals, reg_info,
                    cc_filename,
                    output_contour_plot=confidence_contour_include_ccplot)