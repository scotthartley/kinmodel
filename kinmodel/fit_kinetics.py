#! /usr/bin/env python3
"""Executable script to interface with modules in the kinmodel
package for experimental data fitting.

** Modified by Gyunam Park 24.03.12

"""
import textwrap
import argparse
import kinmodel
import kinmodel.models
import appdirs
import os
import kinmodel.constants as constants


def fit_kinetics():
    model_search_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), constants.MODEL_DIR_NAME),
        os.path.join(appdirs.user_data_dir(constants.APP_NAME,
                                           constants.APP_AUTHOR),
                     constants.MODEL_DIR_NAME),
        os.path.dirname(kinmodel.models.__file__)
        ]

    all_models = kinmodel.KineticModel.get_all_models(model_search_dirs)

    model_help_text = "\nLooking for models in: \n\n{}".format(
            "\n".join(["- " + m for m in model_search_dirs]))

    model_help_text += "\n\nCurrently available models:\n\n"
    for m in sorted(all_models.keys()):
        model_help_text += f"---\n*{m}*\n\n"
        model_help_text += f"Path: {all_models[m].path}\n\n"
        model_help_text += textwrap.indent(
                all_models[m].description, "    ")
        model_help_text += "\n"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=model_help_text)
    parser.add_argument(
        "model_name",
        help="Name of model to apply")
    parser.add_argument(
        "filename",
        help="CSV file to process")
    parser.add_argument(
        "-ks", "--fixed_ks",
        help="Fixed k's and K's for model (will not be optimized).",
        nargs="+", type=float, action = 'append') # New code
    parser.add_argument(
        "-cs", "--fixed_cs",
        help="Fixed concentrations for model (will not be optimized).",
        nargs="+", type=float, action = 'append') # New code
    parser.add_argument(
        "-kg", "--k_guesses",
        help="Override guesses for k's and K's.",
        nargs="+", type=float, action = 'append') # New code
    parser.add_argument(
        "-cg", "--c_guesses",
        help="Override guesses for concentrations.",
        nargs="+", type=float, action = 'append') # New code
    parser.add_argument(
        "-w", "--weight_min_conc",
        help=("Triggers weighted regression relative to the concentration, "
              "with the argument used as the floor (i.e., \"-w 5\" implies "
              "that concentrations will be weighted by 1/c, with 5 as the "
              "min concentration used.)"),
        type=float)
    parser.add_argument(
        "-b", "--bootstrap_iterations",
        help=("Number of bootstrapping iterations to perform for parameter "
              "errors (default=0)"),
        type=int, default=0)
    parser.add_argument(
        "-ci", "--confidence_interval",
        help=("%% confidence interval to use for bootstrap statistics "
              "(default=95)"),
        type=float, default=95)
    parser.add_argument(
        "-bff", "--bootstrap_force1st",
        help=("Force bootstrapping to always use first data points"),
        action='store_true')
    parser.add_argument(
        "-bn", "--bootstrap_nodes",
        help=("Number of nodes to be used in parallel processing "
              "(default=all)"),
        type=int, default=-1)
    parser.add_argument(
        "-cci", "--confidence_contour_intervals",
        help=("Number of intervals to be used in generating confidence "
              "contour plots for pairs of parameters (default=None)"),
        type=int, default=None)
    parser.add_argument(
        "-ccm", "--confidence_contour_multiplier",
        help=("Number of CIs to use around parameters in confidence contour "
              "plots (default=5)"),
        type=float, default=5)
    parser.add_argument(
        "-cccs", "--confidence_contour_concs",
        help=("Include fit concentrations in confidence contours"),
        action='store_true')
    parser.add_argument(
        "-ccicp", "--confidence_contour_include_ccplot",
        help=("Output actual contour plots of confidence contours (and not "
              "just heat maps"),
        action='store_true')
    parser.add_argument(
        "-so", "--summary_output",
        help="Excludes conc vs time data from text output",
        action='store_true')
    parser.add_argument(
        "-tp", "--text_output_points",
        help=("Number of points for curves in text output (not pdf) "
              "(default = 3000)"),
        type=int, default=3000)
    parser.add_argument(
        "-tf", "--text_expansion_factor",
        help=("Expansion factor for curves in text output "
              "(i.e., extension past last point) (default = 3)"),
        type=float, default=3.0)
    parser.add_argument(
        "-ns", "--no_text_save",
        help="Do not save text output (send to stdout instead)",
        action='store_true')
    parser.add_argument(
        "-pp", "--plot_output_points",
        help=("Number of points for curves in output (pdf) "
              "(default = 1000, 0 to disable)"),
        type=int, default=1000)
    parser.add_argument(
        "-pf", "--plot_expansion_factor",
        help=("Expansion factor for curves in output "
              "(i.e., extension past last point) (default = 1.1)"),
        type=float, default=1.1)
    parser.add_argument(
        "-ms", "--more_stats",
        help="Output covariance/correlation matrices",
        action='store_true')
    parser.add_argument(
        "-nv", "--no_verbose",
        help=("Silence verbose output during fit (prints sum square residuals "
              "and bootstrap progress)"),
        action='store_true')
    parser.add_argument(
        "-cy", "--common_y",
        help=("All plots share same max y axis values"),
        action='store_true')
    parser.add_argument(
        "-np", "--no_parameters",
        help=("Do not print parameters on plots"),
        action='store_true')
    parser.add_argument(
        "-u", "--units",
        help=("Time and concentration units, each as a single word"),
        nargs=2, type=str)
    parser.add_argument(
        "-nd", "--no_simulate_direct",
        help=("For indirect fitting, controls whether the direct model is "
              "simulated"),
        action='store_true')
    parser.add_argument(
        "-l", "--load_reg_info",
        help=("Load reg_info from previous optimization; original model must "
              "be specified"),
        action='store_true')
    parser.add_argument(
        "-log","--semilog_x",
        help=("For semilog x plotting"),
        action='store_true') # New code
    args = parser.parse_args()

    model = kinmodel.KineticModel.get_model(args.model_name, all_models)

    if args.weight_min_conc:
        model.weight_func = lambda exp: 1/max(exp, args.weight_min_conc)
        model.description += ("\n\nErrors are weighted by 1/conc, with a "
                              f"min conc of {args.weight_min_conc}")
        model.name += f"_re{args.weight_min_conc}"

    args.fixed_ks=sum(args.fixed_ks, []) if args.fixed_ks else None # New code
    args.fixed_cs=sum(args.fixed_cs, []) if args.fixed_cs else None # New code
    args.k_guesses=sum(args.k_guesses, []) if args.k_guesses else None # New code
    args.c_guesses=sum(args.c_guesses, []) if args.c_guesses else None # New code

    fixed_ks = args.fixed_ks if args.fixed_ks else None
    fixed_cs = args.fixed_cs if args.fixed_cs else None
    k_guesses = args.k_guesses if args.k_guesses else None
    c_guesses = args.c_guesses if args.c_guesses else None

    kinmodel.fit_and_output(
            model=model,
            data_filename=args.filename,
            fixed_ks=fixed_ks,
            fixed_concs=fixed_cs,
            k_guesses=k_guesses,
            conc_guesses=c_guesses,
            text_output_points=args.text_output_points,
            text_time_extension_factor=args.text_expansion_factor,
            text_output=not args.no_text_save,
            plot_output_points=args.plot_output_points,
            plot_time_extension_factor=args.plot_expansion_factor,
            text_full_output=not args.summary_output,
            monitor=not args.no_verbose,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_nodes=args.bootstrap_nodes,
            bootstrap_CI=args.confidence_interval,
            bootstrap_force1st=args.bootstrap_force1st,
            confidence_contour_intervals=args.confidence_contour_intervals,
            confidence_contour_multiplier=args.confidence_contour_multiplier,
            confidence_contour_cs=args.confidence_contour_concs,
            confidence_contour_include_ccplot=args.confidence_contour_include_ccplot,
            more_stats=args.more_stats,
            common_y=args.common_y,
            plot_no_params=args.no_parameters,
            units=args.units,
            simulate=not args.no_simulate_direct,
            load_reg_info=args.load_reg_info,
            plot_semilogx=args.semilog_x # New code
            )
