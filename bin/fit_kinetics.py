#! /usr/bin/env python3
"""Executable script to interface with modules in the kinmodel
package.

"""
import argparse, sys, os, textwrap
import kinmodel

model_help_text = "default models:\n"
for m in kinmodel.default_models:
    model_help_text += f'---\n  *{m}*\n'
    model_help_text += textwrap.indent(kinmodel.default_models[m].description,
            "    ")
    model_help_text += "\n"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=model_help_text)
parser.add_argument("model_name",
    help="Name of model to apply")
parser.add_argument("filename",
    help="CSV file to process")
parser.add_argument("-m", "--new_model",
    help=("Filename of module containing additional model; "
            "must be in working directory, omit .py extension"),
    default=None)
parser.add_argument("-w", "--weight_min_conc",
    help=("Triggers weighted regression relative to the concentration, "
            "with the argument used as the floor (i.e., \"-w 2\" implies "
            "that concentrations will be weighted by 1/c, with 2 as the "
            "min concentration used.)"),
    type=float)
parser.add_argument("-b", "--bootstrap_iterations",
    help=("Number of bootstrapping iterations to perform for parameter errors "
            "(default=100, set to 0 to disable)"),
    type=int, default=0)
parser.add_argument("-ci", "--confidence_interval",
    help=("%% confidence interval to use for bootstrap statistics (default=99)"),
    type=float, default=99)
parser.add_argument("-so", "--summary_output",
    help="Excludes conc vs time data from text output",
    action='store_true')
parser.add_argument("-tp", "--text_output_points",
    help="Number of points for curves in text output (not pdf) (default = 3000)",
    type=int, default=3000)
parser.add_argument("-tf", "--text_expansion_factor",
    help=("Expansion factor for curves in text output "
            "(i.e., extension past last point) (default = 3)"),
    type=float, default=3.0)
parser.add_argument("-ns", "--no_text_save",
    help="Do not save text output (send to stdout instead)",
    action='store_true')
parser.add_argument("-pp", "--plot_output_points",
    help="Number of points for curves in output (pdf) (default = 1000)",
    type=int, default=1000)
parser.add_argument("-pf", "--plot_expansion_factor",
    help=("Expansion factor for curves in output "
            "(i.e., extension past last point) (default = 1.1)"),
    type=float, default=1.1)
parser.add_argument("-np", "--no_plot",
    help="Disable plot output",
    action='store_true')
parser.add_argument("-ms", "--more_stats",
    help="Output covariance/correlation matrices",
    action='store_true')
parser.add_argument("-nv", "--no_verbose",
    help=("Silence verbose output during fit (prints sum square residuals and "
            "bootstrap progress)"),
    action='store_true')
args = parser.parse_args()

sys.path.append(os.getcwd())

if args.new_model:
    new_model = __import__(args.new_model).model
    models = {**kinmodel.default_models, **{new_model.name: new_model}}
else:    
    models = kinmodel.default_models

try:
    model = models[args.model_name]
except KeyError:
    print(f'"{args.model_name}" is not a valid model.')
    print(", ".join(a for a in models), "are currently available.")
    sys.exit(1)

if args.weight_min_conc:
    model.weight_func = lambda exp: 1/max(exp, args.weight_min_conc)
    model.description += ("\n\nErrors are weighted by 1/conc, with a "
            f"min conc of {args.weight_min_conc}")
    model.name += f"_re{args.weight_min_conc}"

kinmodel.fit_and_output(
        model = model, 
        data_filename = args.filename,
        text_output_points = args.text_output_points, 
        text_time_extension_factor = args.text_expansion_factor,
        text_output = not args.no_text_save,
        plot_output_points = args.plot_output_points, 
        plot_time_extension_factor = args.plot_expansion_factor,
        plot_output = not args.no_plot,
        text_full_output = not args.summary_output,
        monitor = not args.no_verbose,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_CI=args.confidence_interval,
        more_stats=args.more_stats)
