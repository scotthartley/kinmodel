kinmodel: A python package for chemical kinetic models
======================================================

kinmodel is a python package for modeling chemical kinetics, and in
particular for fitting kinetic models to experimental concentration vs
time data. It defines the KineticModel class, which contains a system of
differential equations describing a system of chemical reactions and
methods to simulating the system and fitting it to experimental
concentration vs time data. It is a fairly straightforward
implementation of methods in scipy.

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
``fitkinetics.py model_name exp_data.csv``, where ``model_name`` defines
the kinetic model to be applied and ``exp_data.csv`` is a csv file
containing experimental concentration vs time data for relevant species.
The program will output a txt file of the fit parameters and best-fit
curves and a pdf with the plots. At present it produces two plots so
different species can be visualized separately. There are a number of
command line options. Help is available via ``fitkinetics.py -h``.

Some models are included as part of the default installation, but new
ones can be defined in a separate dictionary stored in filename.py and
then loaded with the ``-m filename`` option. This new file should have
the following form:

::

   import textwrap
   from kinmodel import KineticModel

   def test_model(concs, t, *ks):
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
       kin_sys = test_model,
       ks_guesses = [0.02, 0.02, 0.1],
       starting_concs_guesses = [20],
       starting_concs_constant = [0, 0],
       parameter_names = ["k1", "k2", "k3", "[S1]0"],
       legend_names = ["S1", "S2", "S3"],
       top_plot = [1],
       bottom_plot = [2, 3],
       sort_order = [1, 2, 3],
       int_eqn = [],
       int_eqn_desc = []
       )
