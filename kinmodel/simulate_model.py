"""Interfaces with the KineticModel class to simulate conc vs time data
and output the results.

"""
import platform
import numpy as np
import scipy
from matplotlib import pyplot as plt, rcParams
from . import _version

# Parameters and settings for plots.
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
MARKER_SIZE = 6
FIGURE_SIZE_1 = (2.2, 1.9)
FIGURE_SIZE_2 = (2.2, 3.5)
YLABEL = "C"
XLABEL = "t"
rcParams['font.size'] = 6
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['lines.linewidth'] = 0.5
rcParams['axes.linewidth'] = 0.5
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 6

def prepare_text(model, ks, concs, time, num_points, full_output):
    """Generates the output text.

    """
    sim_ts, sim_concs, integrals = model.simulate(ks, concs, num_points, time, 
            integrate=True)

    if len(concs) == model.num_var_concs:
        parameters = ks + concs + model.starting_concs_constant
    elif len(concs) == model.num_concs:
        parameters = ks + concs
    else:
        raise ValueError("Incorrect number of concentrations specified.")

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
    for n in range(len(model.parameter_names)):
        text += (f"{model.parameter_names[n]:>{model.len_params}} = "
                f"{parameters[n]:+.5e}\n")
    text += "\n"
    text += "\n"

    text += "Simulation\n"
    text += "----------\n"
    text += f"Points in simulation: {num_points}\n"
    text += "\n"

    if integrals:
        text += "Integrals:\n"
        text += "\n"
        for n in integrals:
            integral_label = "∫ " + n + " dt"
            text += (f"{integral_label:>{model.len_int_eqn_desc+5}} "
                    f"= {integrals[n]:+.5e}\n")
        text += "\n"


    text += "Concentration Extremes:\n"
    text += "\n"
    for n in range(model.num_concs):
        text += (f"{model.legend_names[n]:>{model.len_legend}} min: "
                f"{sim_concs[:,n].min():+.3e}\n")
        text += (f"{model.legend_names[n]:>{model.len_legend}} max: "
                f"{sim_concs[:,n].max():+.3e}\n")
    text += "\n"

    if full_output:
        text += "Results:\n"
        text += "\n"
        text += "t " + " ".join(model.legend_names) + "\n"    
        for n in range(len(sim_ts)):
            text += str(sim_ts[n]) + " " + " ".join(
                str(m) for m in sim_concs[n]) + "\n"

    return text

def generate_plot(model, ks, concs, time, num_points, output_filename):
    """Generates the output plot.

    Saved as pdf to output_filename.

    """

    if len(concs) == model.num_var_concs:
        parameters = ks + concs + model.starting_concs_constant
    elif len(concs) == model.num_concs:
        parameters = ks + concs
    else:
        raise ValueError("Incorrect number of concentrations specified.")

    smooth_ts_plot, smooth_curves_plot, _ = model.simulate(ks, 
            concs, num_points, time, integrate=False)

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

        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)

        # Print parameters on plot.
        pars_to_print = ""
        for n in range(model.num_ks + model.num_concs):
            pars_to_print += f"{model.parameter_names[n]} = {parameters[n]:.2e}\n"
        plt.text(0.5, 0.2, pars_to_print, transform=plt.gca().transAxes, 
                fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()

def simulate_and_output(model, ks, concs, time, text_num_points, 
        plot_num_points, filename=None, text_full_output=True):
    """Carry out the simulation of the model and output the data.

    """
    
    if text_num_points:
        output_text = prepare_text(model, ks, concs, time, text_num_points, 
                text_full_output)
        if filename:
            with open(f"{filename}.txt", 'w', encoding='utf-8') as write_file:
                print(output_text, file=write_file)
        else:
            print(output_text)

    if plot_num_points and filename:
        generate_plot(model, ks, concs, time, plot_num_points, f"{filename}.pdf")
