import textwrap
from ..KineticModel import KineticModel

def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_2, K1, K2 = ks

    return [ - k1*Ac*E + k_2*An - (k1*Ac**2*E+k_2*An*Ac)/(Ac+K1)
                + (k1*K1*Ac*E+k_2*K1*An)/(Ac+K1) + (2*k1*K2*Ac**2*E+2*k_2*K2*Ac*An)/(Ac+K1),
             - k1*Ac*E,
             + k1*Ac*E,
             - k_2*An + (k1*Ac**2*E)/(Ac+K1) + (k_2*An*Ac)/(Ac+K1) ]

model = KineticModel(
    name = "shared_int_gb_ss",
    description = textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate and direct general
        base catalysis of acylpyridinium intermediate by Ac (i.e., AcO-):

            Ac + E ---> I + U  (k1)
            I + Ac <==> An     (k2, k-2)
                 I ---> Ac     (k3)
            I + Ac ---> 2Ac    (k4)

            Steady-state approximation with K1 = k3/(k2+k4) and K2 = k4/(k2+k4)"""),
    kin_sys = equations,
    ks_guesses = [0.02, 1, 10, 0.01],
    starting_concs_guesses = [50, 50],
    starting_concs_constant = [0, 0],
    parameter_names = ["k1", "k-2", "K1", "K2", "[Acid]0", "[EDC]0", "[U]0", "[An]0"],
    legend_names = ["Acid", "EDC", "Urea", "Anhydride"],
    top_plot = [1, 2],
    bottom_plot = [0, 3],
    sort_order = [1, 3, 2, 0],
    int_eqn = [
    ],
    int_eqn_desc = [
    ]
    )