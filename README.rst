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
``fit_kinetics model_name exp_data.csv``, where ``model_name`` defines
the kinetic model to be applied and ``exp_data.csv`` is a csv file
containing experimental concentration vs time data for relevant species.
The csv file can contain multiple experiments that will be fitted
simultaneously for all rate constants. Experiments are separated by
title lines that must contain a title in column 1 that cannot be
interpreted as a number followed by empty cells for the remaining
columns.

The program will output txt files of the fit parameters and pdf files of
plots. At present, it produces up to two subplots so different species
can be visualized separately. There are a number of command line
options. Help is available via ``fit_kinetics -h``.

Data can also be simulated given a set of parameters using the
model_kinetics.py executable. The format is
``model_kinetics model_name simulation_time par1 par2 par3 ...``.
Default starting concentrations associated with the model can be
overridden by including them at the end of the parameters list. Options
can be specified to control the output. Ranges for the parameters can be
set along with an optional target number of simulations (``-n``): in
this case, the ranges will be divided into equal segments such that the
total number of simulations is at most the target. More information is
available from ``model_kinetics -h``.

Some models are included as part of the default installation, but new
ones can be defined in a separate dictionary stored in filename.py and
then loaded with the ``-m filename`` option. This new file should have
the following form:

::

   import textwrap
   from kinmodel import KineticModel


   def equations(concs, t, *ks):
       Ac, E, U, An = concs
       k1, k_2, K = ks

       return [(- k1*Ac*E - (k1*Ac**2*E)/(Ac+K) - (k_2*An*Ac)/(Ac+K) + k_2*An
                + (k1*K*Ac*E)/(Ac+K) + (k_2*K*An)/(Ac+K)),
               - k1*Ac*E,
               + k1*Ac*E,
               - k_2*An + (k1*Ac**2*E)/(Ac+K) + (k_2*An*Ac)/(Ac+K)]


   model = KineticModel(
       name="shared_int_ss",
       description=textwrap.dedent("""\
           Simple model with shared acylpyridinium intermediate:

               Ac + E ---> I + U  (k1)
               I + Ac <==> An     (k2, k-2)
                    I ---> Ac     (k3)

               Steady-state approximation with K=k3/k2"""),
       kin_sys=equations,
       ks_guesses=[0.02, 0.03, 10],
       ks_constant=[],
       conc0_guesses=[50, 50],
       conc0_constant=[0, 0],
       k_var_names=["k1", "k-2", "K"],
       k_const_names=[],
       conc0_var_names=["[Acid]0", "[EDC]0"],
       conc0_const_names=["[U]0", "[An]0"],
       legend_names=["Acid", "EDC", "Urea", "Anhydride"],
       top_plot=[1, 2],
       bottom_plot=[0, 3],
       sort_order=[1, 3, 2, 0],
       int_eqn=[
               lambda cs, ks: (ks[0]*cs[0]**2*cs[1])/(cs[0]+ks[2]),
               lambda cs, ks: ks[1]*cs[3],
               lambda cs, ks: (ks[1]*cs[3]*cs[0])/(cs[0]+ks[2]), ],
       int_eqn_desc=[
               "(k1*Ac^2*E)/(Ac+K)",
               "k_2*An",
               "(k_2*An*Ac)/(Ac+K)",] ,
       lifetime_conc=[3],
       rectime_conc=[0],
       )

Note that the concentrations (concs) with variable starting
concentration (S1 in the example) passed first, followed by the others.
That is, the total number of entries for starting_concs_guesses and
starting_concs_constant should be equal to the number of species, with
the variable ones always listed first.

Models can also be defined with the IndirectKineticModel class. This
allows normal KineticModel mechanisms to be used in cases where the
experimental observables are a function of the underlying concentrations
(e.g., oligomers where the total functional group concentration is known
but individual concentrations are not). These are defined as in:

::

   import textwrap
   import numpy as np
   from ..KineticModel import IndirectKineticModel


   model = IndirectKineticModel(
       name="DA_explicit_DA2_ss_ind",
       parent_model_name="DA_explicit_DA2_ss",
       description=textwrap.dedent("""\
           Indirect version of the DA_explicit_DA2_ss model, using total
           diacid and total anhydride concentration.\
           """),
       conc_mapping=lambda c: np.array([c[:, 0]+c[:, 3],
                                        c[:, 1],
                                        c[:, 2],
                                        c[:, 3],
                                        c[:, 4]]).transpose(),
       legend_names=["Diacid", "EDC", "Urea", "Linear", "Cyclic"],
       top_plot=[1, 2],
       bottom_plot=[0, 3, 4],
       sort_order=[2, 3, 4, 0, 1],
       int_eqn=[
           ],
       int_eqn_desc=[
           ],
       lifetime_conc=[],
       rectime_conc=[],
       )

Here the parent_model_name defines the underlying mechanism. The
conc_mapping function converts the concentrations of the species into
the experimentally observed quantities. In the example, the “Diacid”
concentration is the sum of the concentrations of species 0 and 4 in the
DA_explicit_DA2 KineticModel.
