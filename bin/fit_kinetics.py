#! /usr/bin/env python3
"""Executable script to interface with modules in the kinmodel
package.

"""
import argparse, sys, os, textwrap
import kinmodel

model_help_text = "default models:\n"
for m in kinmodel.default_models:
    model_help_text += f'  "{m}"\n'
    model_help_text += textwrap.indent(kinmodel.default_models[m].description, "    ")
    model_help_text += "\n\n"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=model_help_text)
parser.add_argument("model_name",
    help="Name of model to apply")
parser.add_argument("filename",
    help="CSV file to process")
parser.add_argument("-m", "--extra_models",
    help="Filename of module containing additional models; must be in working directory, omit .py extension",
    default=None)
parser.add_argument("-tp", "--text_output_points",
    help="Number of points for curves in text output (not pdf) (default = 3000)",
    type=int, default=3000)
parser.add_argument("-tf", "--text_expansion_factor",
    help="Expansion factor for curves in text output (i.e., extend time past last point) (default = 3)",
    type=float, default=3.0)
parser.add_argument("-ns", "--no_text_save",
    help="Do not save text output (send to stdout instead)",
    action='store_true')
parser.add_argument("-so", "--summary_output",
    help="Excludes conc vs time data from text output",
    action='store_true')
parser.add_argument("-pp", "--plot_output_points",
    help="Number of points for curves in output (pdf) (default = 1000)",
    type=int, default=1000)
parser.add_argument("-pf", "--plot_expansion_factor",
    help="Expansion factor for curves in output (i.e., extend time past last point) (default = 1.1)",
    type=float, default=1.1)
parser.add_argument("-np", "--no_plot",
    help="Disable plot output",
    action='store_true')
parser.add_argument("-v", "--verbose",
    help="Verbose output during fit (prints sum square residuals)",
    action='store_true')
args = parser.parse_args()

sys.path.append(os.getcwd())

if args.extra_models:
    extra_models = __import__(args.extra_models).models
    models = {**kinmodel.default_models, **extra_models}
else:    
    models = kinmodel.default_models

try:
    model = models[args.model_name]
except KeyError:
    print(f'"{args.model_name}" is not a valid model.')
    sys.exit(1)

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
    monitor = args.verbose)
