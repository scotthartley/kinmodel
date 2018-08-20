#! /usr/bin/env python3
"""Executable script to simulate kinetic data using the KineticModel
class.

"""
import math
import itertools
import argparse
import kinmodel

PAR_ERR_TEXT = "Invalid parameter input format"
RANGE_IND = ".."


def model_kinetics():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        help=("Name of model to use (see fit_kinetics.py -h for list of "
              "default models)"))
    parser.add_argument(
        "time",
        help="Total time for simulation",
        type=float)
    parser.add_argument(
        "parameters",
        help="List of parameters for model",
        nargs="+")
    parser.add_argument(
        "-f", "--filename",
        help="Root filename for output (no extension)")
    parser.add_argument(
        "-n", "--sim_number",
        help="Number of simulations (for parameter ranges, default 2^n)",
        type=int)
    parser.add_argument(
        "-m", "--new_model",
        help=("Filename of module containing additional models; "
              "must be in working directory, omit .py extension"),
        default=None)
    parser.add_argument(
        "-tp", "--text_output_points",
        help=("Number of points for curves in text output (not pdf) "
              "(default = 3000)"),
        type=int, default=3000)
    parser.add_argument(
        "-so", "--summary_output",
        help="Excludes conc vs time data from text output",
        action='store_true')
    parser.add_argument(
        "-pp", "--plot_output_points",
        help="Number of points for curves in output (pdf) (default = 1000)",
        type=int, default=1000)
    parser.add_argument(
        "-u", "--units",
        help=("Time and concentration units, each as a single word"),
        nargs=2, type=str)
    args = parser.parse_args()

    num_ranges = len([t for t in args.parameters if RANGE_IND in t])
    if args.sim_number:
        sim_number = args.sim_number
    else:
        sim_number = 2**(num_ranges)

    if num_ranges > 0:
        sims_per_range = math.floor(sim_number**(1/num_ranges))
        index_digits = math.floor(math.log10(sims_per_range**num_ranges)) + 1
    else:
        sims_per_range = None
        index_digits = 1

    if sims_per_range == 1:
        raise ValueError("Too few simulations specified for number of ranged "
                         "parameters.")

    parameters = []
    for parameter in args.parameters:
        parameter_range = parameter.split(RANGE_IND)
        if len(parameter_range) == 1:
            try:
                parameters.append([float(parameter_range[0])])
            except ValueError:
                print(PAR_ERR_TEXT)
        elif len(parameter_range) == 2:
            try:
                p0 = float(parameter_range[0])
                p1 = float(parameter_range[1])
                delta = (p1-p0)/(sims_per_range-1)
                parameters.append([])
                for n in range(sims_per_range):
                    parameters[-1].append(p0 + delta*n)
            except ValueError:
                print(PAR_ERR_TEXT)
        else:
            raise ValueError(PAR_ERR_TEXT)

    model = kinmodel.KineticModel.get_model(args.model_name, args.new_model)

    set_num = 0
    for parameter_set in itertools.product(*parameters):
        set_num += 1
        ks = list(parameter_set[:model.num_ks])
        concs = list(parameter_set[model.num_ks:])
        if args.filename:
            filename = args.filename + f"_{set_num:0{index_digits}}"
        else:
            filename = None
        kinmodel.simulate_and_output(
                model=model,
                ks=ks,
                concs=concs,
                time=args.time,
                text_num_points=args.text_output_points,
                plot_num_points=args.plot_output_points,
                filename=filename,
                text_full_output=not args.summary_output,
                units=args.units)
