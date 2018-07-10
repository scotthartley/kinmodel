"""Interfaces with the KineticModel class to fit experimental data to a
given kinetic model.

"""
import platform
import numpy as np, scipy
from matplotlib import pyplot as plt, rcParams

# Parameters and settings for plots.
COLORS = ['b','g','r','c','m','y','k']
MARKER_SIZE = 6
rcParams['font.size'] = 6
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['lines.linewidth'] = 0.5
rcParams['axes.linewidth'] = 0.5
rcParams['legend.frameon'] = False
rcParams['legend.fontsize'] = 6
plt.figure(figsize=(2.2,3.5))

# Prevent line breaking and format numbers from np
np.set_printoptions(linewidth=np.nan)
np.set_printoptions(precision=2,suppress=True)


def get_raw_data(model, data_filename):
    """Load data from file, formated as a csv file. The file is assumed
    to include a header row, and the order of columns must match that
    specified by the model (model.sort_order).

    Returns the experimental times (np array), data (np array), and
    total number of data points (int)

    """
    with open(data_filename) as datafile:
        next(datafile)  # Skip header
        ts = []         # List of experimental times
        raw_data = []       # List of lists of experimental concentrations
        total_points = 0         # Total number of experimental points
        for line in datafile:
            curline = line.replace("\n", "").split(',')
            ts.append(float(curline[0]))
            concs = []

            for n in range(model.num_concs):
                if n+1 < len(curline):
                    if curline[n+1] != '':
                        total_points += 1
                        concs.append(float(curline[n+1]))
                    else:
                        concs.append(np.nan)
                else:
                    concs.append(np.nan)

            raw_data.append(concs)

        unsorted_data = np.array(raw_data)

        sorted_data = np.empty_like(unsorted_data)
        for n in range(model.num_concs):
            sorted_data[:,n] = unsorted_data[:,model.sort_order[n]]

    return np.array(ts), sorted_data, total_points


def prepare_text(model, fit_ks, fit_concs, reg_info, num_points, max_time, 
    filename, full_simulation=True):
    """Generates the output text. The total time (max_time) and number
    of points (numpoints) for the output simulation must be specified.
    Output of the full simulation can be controlled.

    """
    smooth_ts_out, smooth_curves_out, integrals = model.simulate(fit_ks, 
        fit_concs, num_points, max_time)

    all_params = fit_ks + fit_concs

    title = f"Regression results for file \"{filename}\"" 
    text = title + "\n"
    text += "="*len(title) + "\n"
    text += "\n"

    text += f"Python version: {platform.python_version()}\n"
    text += f"Numpy version: {np.version.version}\n"
    text += f"Scipy version: {scipy.version.version}\n"
    text += "\n"

    text += "Model\n"
    text += "-----\n"
    text += f"Name: {model.name}\n"
    text += f"Description: {model.description}\n"
    text += "\n"

    text += "Optimized parameters\n"
    text += "--------------------\n"
    for n in range(len(all_params)):
        text += f"{model.parameter_names[n]:>{model.len_params}} = {all_params[n]:+5e}\n"
    text += "\n"

    if integrals:
        text += "Integrals\n"
        text += "---------\n"
        for n in integrals:
            integral_label = "∫ "+n+" dt"
            text += f"{integral_label:>{model.len_int_eqn_desc+5}} = {integrals[n]:+5e}\n"
        text += "\n"

    text += "Regression info\n"
    text += "---------------\n"

    text += f"Success: {reg_info['success']}\n"
    text += f"Msg: {reg_info['message']}\n"

    text += f"Total points (dof): {reg_info['total_points']} ({reg_info['dof']})\n"
    text += f"Sum square errors: {reg_info['sse']}\n"
    text += f"Std Deviation of errors: {reg_info['sde']}\n"
    text += "\n"

    if full_simulation:
        text += "Results\n"
        text += "-------\n"
        text += "t " + " ".join(model.legend_names) + "\n"
        
        for n in range(len(smooth_ts_out)):
            text += str(smooth_ts_out[n]) + " " + " ".join(
                str(m) for m in smooth_curves_out[n]) + "\n"

    return text


def generate_plot(model, fit_ks, fit_concs, num_points, max_time, exp_times, 
    exp_concs, output_filename):
    """Generates the output plot. Number of points and maximum time must be
    specified. Saved as pdf to output_filename.

    """
    all_params = fit_ks + fit_concs
    smooth_ts_plot, smooth_curves_plot, _ = model.simulate(fit_ks, 
        fit_concs, num_points, max_time, integrate=False)

    # Plot the data and save as pdf.
    plt.subplot(211)
    col = 0
    for n in [np.array(exp_concs).T[n] for n in model.top_plot]:
        plt.scatter(exp_times, n, c=COLORS[col], s=MARKER_SIZE, linewidths=0)
        col += 1

    col = 0
    for n in [smooth_curves_plot.T[n] for n in model.top_plot]:
        plt.plot(smooth_ts_plot, n, COLORS[col] + '-')
        col += 1

    plt.legend([model.legend_names[n] for n in model.top_plot], loc=4)

    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=smooth_ts_plot[-1])

    plt.ylabel("C (mM)")

    # Print parameters on plot.
    pars_to_print = ""
    for n in range(len(all_params)):
        pars_to_print += "{} = {:.2e}\n".format(model.parameter_names[n], 
            all_params[n])
    plt.text(0.5, 0.2, pars_to_print, transform=plt.gca().transAxes, fontsize=6)
    
    plt.subplot(212)
    col = 0
    for n in [np.array(exp_concs).T[n] for n in model.bottom_plot]:
        plt.scatter(exp_times, n, c=COLORS[col], s=MARKER_SIZE, 
            linewidths=0, zorder=2)
        col += 1

    col = 0
    for n in [smooth_curves_plot.T[n] for n in model.bottom_plot]:
        plt.plot(smooth_ts_plot, n, COLORS[col] + '-', zorder=3)
        col += 1

    plt.legend([model.legend_names[n] for n in model.bottom_plot], loc=2)

    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=smooth_ts_plot[-1])

    plt.xlabel('t (min)')
    plt.ylabel('C (mM)')

    plt.tight_layout()

    plt.savefig(output_filename)


def fit_and_output(model, data_filename,
    text_output_points=3000, text_time_extension_factor=3.0, text_output=True, 
    plot_output_points=1000, plot_time_extension_factor=1.1, plot_output=True,
    text_full_output=True, monitor=False):
    """Carry out the fit of the model and output the data.

    """
    # Get data from input file.
    exp_times, exp_concs, total_points = get_raw_data(model, data_filename)

    # Carry out the fit.
    fit_ks, fit_concs, reg_info = model.fit_to_model(exp_times, exp_concs, 
        total_points, monitor=monitor)

    # Prepare output text and either save to a file or print to stdout.
    output_text = prepare_text(model, fit_ks, fit_concs, reg_info, 
        text_output_points, max(exp_times)*text_time_extension_factor, 
        data_filename, text_full_output)
    if text_output:
        with open(f"{data_filename}_{model.name}.txt", 'w') as write_file:
            print(output_text, file=write_file)
    else:
        print(output_text)

    # Generate plot.
    if plot_output:
        generate_plot(model, fit_ks, fit_concs, plot_output_points, 
            max(exp_times)*plot_time_extension_factor, exp_times, exp_concs, 
            data_filename + f"_{model.name}.pdf")
