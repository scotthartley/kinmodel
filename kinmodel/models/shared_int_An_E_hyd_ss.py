import textwrap
from ..KineticModel import KineticModel

def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_2, k4, k5, K = ks

    return [ - k1*Ac*E - (k1*Ac**2*E)/(Ac+K) - (k_2*An*Ac)/(Ac+K) + k_2*An 
                + (k1*K*Ac*E)/(Ac+K) + (k_2*K*An)/(Ac+K) + 2*k4*An,
             - k1*Ac*E - k5*E,
             + k1*Ac*E + k5*E,
             - k_2*An + (k1*Ac**2*E)/(Ac+K) + (k_2*An*Ac)/(Ac+K) - k4*An]

model = KineticModel(
    name = "shared_int_An_E_hyd_ss",
    description = textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate, direct anhydride 
        hydrolysis, and direct EDC hydrolysis:

            Ac + E ---> I + U  (k1)
            I + Ac <==> An     (k2, k-2)
                 I ---> Ac     (k3)
                An ---> 2Ac    (k4)
                 E ---> U      (k5)

            Steady-state approximation with K = k3/k2"""),
    kin_sys = equations,
    ks_guesses = [0.02, 0.03, 10, 0.1, 10],
    starting_concs_guesses = [50, 50],
    starting_concs_constant = [0, 0],
    parameter_names = ["k1", "k-2", "k4", "k5", "K", "[Acid]0", "[EDC]0"],
    legend_names = ["Acid", "EDC", "Urea", "Anhydride"],
    top_plot = [1, 2],
    bottom_plot = [0, 3],
    sort_order = [1, 3, 2, 0],
    int_eqn = [
    ],
    int_eqn_desc = [
    ]
    )