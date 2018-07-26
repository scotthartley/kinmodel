import textwrap
from ..KineticModel import KineticModel

def equations(concs, t, *ks):
    Ac, E, U, An, I = concs
    k1, k2, k_2, k3 = ks

    return [ - k1*Ac*E - k2*Ac*I + k_2*An + k3*I,
             - k1*Ac*E,
             + k1*Ac*E,
             - k_2*An + k2*I*Ac,
             + k1*Ac*E - k2*I*Ac + k_2*An - k3*I]

model = KineticModel(
    name = "shared_int",
    description = textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate:

            Ac + E ---> I + U  (k1)
            I + Ac <==> An     (k2, k-2)
                 I ---> Ac     (k3)

            No steady-state approximation"""),
    kin_sys = equations,
    ks_guesses = [0.02, 0.03, 0.1, 1],
    starting_concs_guesses = [50, 50],
    starting_concs_constant = [0, 0, 0],
    parameter_names = ["k1", "k2", "k-2", "k3", "[Acid]0", "[EDC]0"],
    legend_names = ["Acid", "EDC", "Urea", "Anhydride", "Int"],
    top_plot = [1, 2],
    bottom_plot = [0, 3, 4],
    sort_order = [1, 3, 2, 0, 4],
    int_eqn = [
        lambda cs, ks: ks[2]*cs[3],
        lambda cs, ks: ks[1]*cs[4]*cs[0],
    ],
    int_eqn_desc = [
        "k_2*An",
        "k2*I*Ac",
    ]
    )