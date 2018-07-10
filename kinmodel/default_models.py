"""Defines the default kinetic models included with the kinmodel
package. Each includes two things:

1. A function that sets up a list of differential equations to be
   solved. The equations should be in the same order as the list of
   concentrations in the parameters. Concentrations should begin with
   ones for which the starting concentration will be fit.

2. A definition in the dictionary that sets up parameters and guesses
   for the model. This definition can also include terms that can be
   integrated, along with their descriptions.

"""
import numpy as np
import textwrap
from .KineticModel import KineticModel

# These functions define the differential equations used in the
# KineticModel instances.
def common_int_ss(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_2, K = ks

    return [ - k1*Ac*E - (k1*Ac**2*E)/(Ac+K) - (k_2*An*Ac)/(Ac+K) + k_2*An 
                + (k1*K*Ac*E)/(Ac+K) + (k_2*K*An)/(Ac+K),
             - k1*Ac*E,
             + k1*Ac*E,
             - k_2*An + (k1*Ac**2*E)/(Ac+K) + (k_2*An*Ac)/(Ac+K) ]

def simple_monoacid(concs, t, *ks):
    Ac, E, U, An = concs
    k1, K, k2 = ks

    return [ - k1*E*Ac + 2*k2*An + (k1*K*Ac*E)/(K+Ac) - (k1*E*Ac**2)/(K+Ac),
             - k1*E*Ac,
             + k1*E*Ac, 
             + (k1*E*Ac**2)/(K+Ac) - k2*An ]


default_models = {
    'common_int_ss': KineticModel(
        name = "common_int_ss",
        description = textwrap.dedent("""\
            Simple model with shared acylpyridinium intermediate:

                Ac + E ---> I + U (k1)
                I + Ac <==> An    (k2, k-2)
                I      ---> Ac    (k3)

                Steady-state approximation with K = k3/k2"""),
        kin_sys = common_int_ss,
        ks_guesses = [0.02, 0.03, 10],
        starting_concs_guesses = [50, 50],
        starting_concs_constant = [0, 0],
        parameter_names = ["k1", "k-2", "K", "[Acid]0", "[EDC]0"],
        legend_names = ["Acid", "EDC", "Urea", "Anhydride"],
        top_plot = [1, 2],
        bottom_plot = [0, 3],
        sort_order = [1, 3, 2, 0],
        int_eqn = [
            lambda cs, ks: (ks[0]*cs[0]**2*cs[1])/(cs[0]+ks[2]),
            lambda cs, ks: ks[1]*cs[3],
            lambda cs, ks: (ks[1]*cs[3]*cs[0])/(cs[0]+ks[2]),
        ],
        int_eqn_desc = [
            "(k1*Ac^2*E)/(Ac+K)",
            "k_2*An",
            "(k_2*An*Ac)/(Ac+K)",
        ]
        ),
    
    'simple_monoacid': KineticModel(
        name="simple_monoacid",
        description=textwrap.dedent("""\
            Simplest monoacid kinetic model:

                Ac + E ---> I      (k1)
                I + Ac ---> An + U (kiAn)
                I      ---> Ac + U (kiAc)
                An     ---> 2Ac    (k2)

                Steady-state approximation with K = kiAc/kiAn"""),
        kin_sys = simple_monoacid,
        ks_guesses = [0.02, 200, 0.5],
        starting_concs_guesses = [100, 400],
        starting_concs_constant = [0, 0],
        parameter_names = ["k1", "K", "k2", "[Acid]0", "[EDC]0"],
        legend_names = ["Acid", "EDC", "Urea", "Anhydride"],
        top_plot = [1, 2],
        bottom_plot = [0, 3],
        sort_order = [1, 3, 2, 0],
        int_eqn = [
            lambda cs, ks: ks[2]*cs[3],
        ],
        int_eqn_desc = [
            "k_2*An",
        ]
        ),
}
