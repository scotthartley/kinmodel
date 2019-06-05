"""Interfaces with the KineticModel class to fit experimental data to a
given kinetic model and output the results.

"""
import platform
import numpy as np
import scipy
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

PLT_PARAMS = {'font.size': 8,
              'font.family': 'sans-serif',
              'font.sans-serif': ['Arial'],
              'lines.linewidth': 0.75,
              'axes.linewidth': 0.75,
              'legend.frameon': False,
              'legend.fontsize': 6}

# Prevent line breaking and format numbers from np
np.set_printoptions(linewidth=np.nan)
np.set_printoptions(precision=2, suppress=True)


def prepare_text(
        model, reg_info, ds_num, num_points, time_exp_factor, filename="",
        boot_CI=95, full_simulation=True, more_stats=False):
    """Generates the output text.

    The number of points (num_points) for the output simulation and
    integrals must be specified.

    """
    smooth_ts_out, smooth_curves_out, integrals = model.simulate(
            reg_info['fit_ks'] + reg_info['fixed_ks'],
            reg_info['fit_concs'][ds_num] + reg_info['fixed_concs'][ds_num],
            num_points,
            time_exp_factor*max(reg_info['dataset_times'][ds_num]))

    num_ks = len(reg_info['fit_ks'])
    num_concs = len(reg_info['fit_concs'][ds_num])
    # Starting index for dataset-specific concentration parameters.
    conc0_i = num_ks + num_concs*ds_num

    dataset_params = reg_info['fit_ks'] + reg_info['fit_concs'][ds_num]
    dataset_consts = reg_info['fixed_ks'] + reg_info['fixed_concs'][ds_num]

    cov_stddevs = (
            reg_info['cov_errors'][:num_ks].tolist()
            + reg_info['cov_errors'][conc0_i:conc0_i+num_concs].tolist())
    if 'boot_num' in reg_info and boot_CI:
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
    if 'boot_num' in reg_info and boot_CI:
        for n in range(len(dataset_params)):
            text += (f"{model.parameter_names[n]:>{model.len_params}} "
                     f"= {dataset_params[n]:+5e} "
                     f"± {(param_CIs[1][n]-param_CIs[0][n])/2:.1e} "
                     f"({param_CIs[0][n]:+5e}, {param_CIs[1][n]:+5e})\n")
        text += "\n"
        text += (f"Errors are {boot_CI}% confidence intervals from "
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
    text += f"Sum square residuals (weighted): {reg_info['ssr']:.2e}\n"
    text += f"Sum square residuals (unweighted): {reg_info['pure_ssr']:.2e}\n"
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
        for n in integrals:
            integral_label = "∫ " + n + " dt"
            text += (f"{integral_label:>{model.len_int_eqn_desc+5}} "
                     f"= {integrals[n]:+5e}\n")
        text += "\n"

    if full_simulation:
        text += "Results:\n"
        text += "\n"
        if 'boot_num' in reg_info and boot_CI:
            assert (list(reg_info['boot_plot_ts'][ds_num])
                    == list(smooth_ts_out)), ("Simulation and bootstrap"
                                              "points do not line up!")
            boot_CI_data = reg_info['boot_plot_CIs'][ds_num]
            text += ("t " + " ".join(model.legend_names) + " "
                     + " ".join(f"{n}CI− {n}CI+" for n in model.legend_names)
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
                  output_filename, boot_CI=95, common_y=True, units=None):
    """Generates the output plot.

    Number of points must be specified. Saved as pdf to output_filename.

    """
    rcParams.update(PLT_PARAMS)

    dataset_params = reg_info['fit_ks'] + reg_info['fit_concs'][ds_num]
    dataset_consts = reg_info['fixed_ks'] + reg_info['fixed_concs'][ds_num]

    max_time = max(reg_info['dataset_times'][ds_num])*time_exp_factor

    smooth_ts_plot, smooth_curves_plot, _ = model.simulate(
            reg_info['fit_ks'] + reg_info['fixed_ks'],
            reg_info['fit_concs'][ds_num] + reg_info['fixed_concs'][ds_num],
            num_points, max_time, integrate=False)

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
                    max(reg_info['max_exp_concs'][n] for n in model.top_plot),
                    max(reg_info['max_pred_concs'][n] for n in model.top_plot)
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

        # Print parameters on plot.
        pars_to_print = ""
        for n in range(model.num_params):
            pars_to_print += "{} = {:.2e}\n".format(model.parameter_names[n],
                                                    dataset_params[n])
        for n in range(model.num_consts):
            pars_to_print += "{} = {:.2e}\n".format(model.constant_names[n],
                                                    dataset_consts[n])

        plt.text(PARAM_LOC[0], PARAM_LOC[1], pars_to_print, transform=plt.gca().transAxes,
                 fontsize=rcParams['legend.fontsize'])

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def fit_and_output(
            model, data_filename, fixed_ks=None, fixed_concs=None,
            k_guesses=None, conc_guesses=None, text_output_points=3000,
            text_time_extension_factor=3.0, text_output=True,
            plot_output_points=1000, plot_time_extension_factor=1.1,
            text_full_output=True, monitor=False,
            bootstrap_iterations=100, bootstrap_CI=95,
            bootstrap_force1st=False, bootstrap_nodes=None, more_stats=False,
            common_y=True, units=None, simulate=True):
    """Carry out the fit of the model and output the data.

    """
    datasets = Dataset.read_raw_data(model, data_filename)

    reg_info = model.fit_to_model(
            datasets, ks_guesses=k_guesses,
            conc0_guesses=conc_guesses, ks_const=fixed_ks,
            conc0_const=fixed_concs, monitor=monitor,
            N_boot=bootstrap_iterations, boot_CI=bootstrap_CI,
            boot_points=text_output_points,
            boot_t_exp=text_time_extension_factor,
            boot_force1st=bootstrap_force1st,
            boot_nodes=bootstrap_nodes)

    for n in range(reg_info['num_datasets']):
        output_text = prepare_text(
                model, reg_info, n, text_output_points,
                text_time_extension_factor, data_filename, bootstrap_CI,
                text_full_output, more_stats)
        if text_output:
            text_filename = (f"{data_filename}_{model.name}"
                             f"_{reg_info['dataset_names'][n]}.txt")
            with open(text_filename, 'w', encoding='utf-8') as write_file:
                print(output_text, file=write_file)
        else:
            print(output_text)

    if plot_output_points:
        for n in range(reg_info['num_datasets']):
            plot_filename = (f"{data_filename}_{model.name}_"
                             f"{reg_info['dataset_names'][n]}.pdf")
            generate_plot(model, reg_info, n, plot_output_points,
                          plot_time_extension_factor, plot_filename,
                          bootstrap_CI, common_y, units)

    if (type(model) is IndirectKineticModel) and simulate:
        for n in range(reg_info['num_datasets']):
            sim_filename = (f"{data_filename}_{model.name}_sim_"
                            f"{reg_info['dataset_names'][n]}")
            simulate_and_output(
                    model=model.parent_model,
                    ks=reg_info['fit_ks'] + reg_info['fixed_ks'],
                    concs=(reg_info['fit_concs'][n]
                           + reg_info['fixed_concs'][n]),
                    time=(max(reg_info['dataset_times'][n])
                          * text_time_extension_factor),
                    text_num_points=text_output_points,
                    plot_num_points=plot_output_points,
                    filename=sim_filename,
                    text_full_output=True,
                    units=units,
                    plot_time=(max(reg_info['dataset_times'][n])
                               * plot_time_extension_factor))
