"""Interfaces with the KineticModel class to simulate conc vs time data
and output the results.

** Modified by Gyunam Park 24.02.28

"""
import platform
import numpy as np
import scipy
from matplotlib import pyplot as plt, rcParams
from . import _version

# Parameters and settings for plots.
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
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


def _resolve_parameters(model, ks, concs):
    if len(ks) == model.num_var_ks:
        all_ks = ks + model.ks_constant
    elif len(ks) == model.num_ks:
        all_ks = ks
    else:
        raise ValueError("Incorrect number of k's specified.")

    if len(concs) == model.num_var_concs0:
        all_concs = concs + model.conc0_constant
    elif len(concs) == model.num_concs0:
        all_concs = concs
    else:
        raise ValueError("Incorrect number of concentrations specified.")

    return all_ks, all_concs


def prepare_text(model, ks, concs, time, num_points, full_output):
    """Generates the output text.

    """
    sim_ts, sim_concs, integrals, calculations = model.simulate(
            ks, concs, time, integrate=True, calcs=True) # change argument

    all_ks, all_concs = _resolve_parameters(model, ks, concs)

    title = f"Simulation of model {model.name}"

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

    text += "Parameters\n"
    text += "----------\n"
    width = max(model.len_params, model.len_consts)
    for n in range(model.num_var_ks):
        text += (f"{model.k_var_names[n]:>{width}} = "
                 f"{all_ks[n]:+.5e}\n")
    for n in range(model.num_const_ks):
        text += (f"{model.k_const_names[n]:>{width}} = "
                 f"{all_ks[model.num_var_ks+n]:+.5e}\n")
    for n in range(model.num_var_concs0):
        text += (f"{model.conc0_var_names[n]:>{width}} = "
                 f"{all_concs[n]:+.5e}\n")
    for n in range(model.num_const_concs0):
        text += (f"{model.conc0_const_names[n]:>{width}} = "
                 f"{all_concs[model.num_var_concs0+n]:+.5e}\n")

    text += "\n"
    text += "\n"

    text += "Simulation\n"
    text += "----------\n"
    text += f"Points in simulation: {num_points}\n"
    text += f"Total time: {time}\n"
    text += "\n"

    if integrals:
        text += "Integrals:\n"
        text += "\n"
        for integral in integrals:
            integral_label = "∫ " + integral[0] + " dt"
            text += (f"{integral_label:>{model.len_int_eqn_desc+5}} "
                     f"= {integral[1]:+.5e}\n")
        text += "\n"

    if calculations:
        text += "Calculations:\n"
        text += "\n"
        for calc in calculations:
            calc_label = calc[0]
            text += (f"{calc_label:>{model.len_calcs_desc}} "
                     f"= {calc[1]:+5e}\n")
        text += "\n"

    text += "Concentration extremes:\n"
    text += "\n"
    for n in range(model.num_concs0):
        text += (f"{model.legend_names[n]:>{model.len_legend}} min: "
                 f"{sim_concs[:,n].min():+.3e}\n")
        text += (f"{model.legend_names[n]:>{model.len_legend}} max: "
                 f"{sim_concs[:,n].max():+.3e}\n")
    text += "\n"

    if model.lifetime_conc:
        text += "Lifetimes:\n"
        text += "\n"
        for n in model.lifetime_conc:
            conc_list = sim_concs[:,n]
            max_conc = conc_list.max()
            max_ind = conc_list.argmax()
            for f in model.lifetime_fracs:
                target_conc = max_conc*f
                target_ind = (np.abs(conc_list[max_ind:] -
                              target_conc).argmin() + max_ind)
                target_time = sim_ts[target_ind]
                text += (f"{model.legend_names[n]:>{model.len_legend}} t to "
                         f"{f*100:6.2f}% of max: {target_time:+.3e}\n")
        text += "\n"

    if model.rectime_conc:
        text += "Recovery times:\n"
        text += "\n"
        for n in model.rectime_conc:
            conc_list = sim_concs[:, n]
            initial_conc = conc_list[0]
            min_ind = conc_list.argmin()
            for f in model.rectime_fracs:
                target_conc = initial_conc*f
                target_ind = (np.abs(conc_list[min_ind:] -
                              target_conc).argmin() + min_ind)
                target_time = sim_ts[target_ind]
                text += (f"{model.legend_names[n]:>{model.len_legend}} t to "
                         f"{f*100:6.2f}% of initial: {target_time:+.3e}\n")
        text += "\n"

    if full_output:
        text += "Results:\n"
        text += "\n"
        text += "t " + " ".join(model.legend_names) + "\n"
        for n in range(len(sim_ts)):
            text += str(sim_ts[n]) + " " + " ".join(
                str(m) for m in sim_concs[n]) + "\n"

    return text


def generate_plot(model, ks, concs, time, num_points, output_filename,
                  units=None):
    """Generates the output plot.

    Saved as pdf to output_filename.

    """
    rcParams.update(PLT_PARAMS)

    smooth_ts_plot, smooth_curves_plot, _, _ = model.simulate(
            ks, concs, time, integrate=False, calcs=False) # change argument

    all_ks, all_concs = _resolve_parameters(model, ks, concs)

    if model.top_plot:
        plt.figure(figsize=FIGURE_SIZE_2)
    else:
        plt.figure(figsize=FIGURE_SIZE_1)

    if model.top_plot:
        plt.subplot(211)
        col = 0
        for n in [smooth_curves_plot.T[m] for m in model.top_plot]:
            plt.plot(smooth_ts_plot, n, COLORS[col] + '-')
            col += 1

        plt.legend([model.legend_names[n] for n in model.top_plot], loc=4)

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
        for n in [smooth_curves_plot.T[n] for n in model.bottom_plot]:
            plt.plot(smooth_ts_plot, n, COLORS[col] + '-', zorder=3)
            col += 1

        plt.legend([model.legend_names[n] for n in model.bottom_plot], loc=2)

        plt.ylim(ymin=0)
        plt.xlim(xmin=0, xmax=smooth_ts_plot[-1])

        if units:
            plt.xlabel(f"{XLABEL} ({units[0]})")
            plt.ylabel(f"{YLABEL} ({units[1]})")
        else:
            plt.xlabel(XLABEL)
            plt.ylabel(YLABEL)

        pars_to_print = ""
        for n in range(model.num_var_ks):
            pars_to_print += (f"{model.k_var_names[n]} = "
                              f"{all_ks[n]:+.2e}\n")
        for n in range(model.num_const_ks):
            pars_to_print += (f"{model.k_const_names[n]} = "
                              f"{all_ks[model.num_var_ks+n]:+.2e}\n")
        for n in range(model.num_var_concs0):
            pars_to_print += (f"{model.conc0_var_names[n]} = "
                              f"{all_concs[n]:+.2e}\n")
        for n in range(model.num_const_concs0):
            pars_to_print += (f"{model.conc0_const_names[n]} = "
                              f"{all_concs[model.num_var_concs0+n]:+.2e}\n")
        plt.text(PARAM_LOC[0], PARAM_LOC[1], pars_to_print, transform=plt.gca().transAxes,
                 fontsize=rcParams['legend.fontsize'])

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def simulate_and_output(model, ks, concs, time, text_num_points,
                        plot_num_points, filename=None, text_full_output=True,
                        units=None, plot_time=None):
    """Carry out the simulation of the model and output the data.

    """

    if not plot_time.any():
        plot_time = time

    if text_num_points:
        output_text = prepare_text(model, ks, concs, time, text_num_points,
                                   text_full_output)
        if filename:
            with open(f"{filename}.txt", 'w', encoding='utf-8') as write_file:
                print(output_text, file=write_file)
        else:
            print(output_text)

    if plot_num_points and filename:
        generate_plot(model, ks, concs, plot_time, plot_num_points,
                      f"{filename}.pdf", units)
