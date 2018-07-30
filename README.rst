kinmodel: A python package for chemical kinetic models
======================================================

kinmodel is a python package for modeling chemical kinetics, and in
particular for fitting kinetic models to experimental concentration vs
time data. It defines the KineticModel class, which contains a system of
differential equations describing a system of chemical reactions and
methods for simulating the system and fitting it to experimental
concentration vs time data. It is a fairly straightforward
implementation of methods in Scipy.

Requirements
------------

-  Python 3.6
-  Numpy
-  Scipy
-  Matplotlib

Installation
------------

The package can be installed as usual with pip.

Use
---

At this point, primary interaction with kinmodel will probably be
through the fit_kinetics.py executable. Basic usage is of the form
``fit_kinetics.py model_name exp_data.csv``, where ``model_name``
defines the kinetic model to be applied and ``exp_data.csv`` is a csv
file containing experimental concentration vs time data for relevant
species. The csv file can contain multiple experiments that will be
fitted simultaneously for all rate constants. Experiments are separated
by title lines that must contain a title in column 1 that cannot be
interpreted as a number followed by empty cells for the remaining
columns.

The program will output txt files of the fit parameters and pdf files of
plots. At present, it produces up to two subplots so different species
can be visualized separately. There are a number of command line
options. Help is available via ``fit_kinetics.py -h``.

Data can also be simulated given a set of parameters using the
model_kinetics.py executable. The format is
``model_kinetics.py model_name simulation_time par1 par2 par3 ...``.
Default starting concentrations associated with the model can be
overridden by including them at the end of the parameters list. Options
can be specified to control the output. Ranges for the parameters can be
set along with an optional target number of simulations (``-n``): in
this case, the ranges will be divided into equal segments such that the
total number of simulations is at most the target. More information is
available from ``model_kinetics.py -h``.

Some models are included as part of the default installation, but new
ones can be defined in a separate dictionary stored in filename.py and
then loaded with the ``-m filename`` option. This new file should have
the following form:

::

   import textwrap
   from kinmodel import KineticModel

   def equations(concs, t, *ks):
       S1, S2, S3 = concs
       k1, k2, k3 = ks

       return [ - k1*S1 + k2S2,
                + k1*S1 - k2*S2 - k3*S3,
                + k3*S2 ]

   model = KineticModel(
       name = "test_model",
       description = textwrap.dedent("""\
           Simple kinetic model:

               S1 <==> S2 (k1, k2)
               S2 ---> S3 (k3)"""),
       kin_sys = equations,
       ks_guesses = [0.02, 0.02, 0.1],
       starting_concs_guesses = [20],
       starting_concs_constant = [0, 0],
       parameter_names = ["k1", "k2", "k3", "[S1]0", "[S2]0", "[S3]0"],
       legend_names = ["S1", "S2", "S3"],
       top_plot = [1],
       bottom_plot = [2, 3],
       sort_order = [1, 2, 3],
       int_eqn = [],
       int_eqn_desc = [],
       )

Note that the concentrations (concs) with variable starting
concentration (S1 in the example) passed first, followed by the others.
That is, the total number of entries for starting_concs_guesses and
starting_concs_constant should be equal to the number of species, with
the variable ones always listed first.
