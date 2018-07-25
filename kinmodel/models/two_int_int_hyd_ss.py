import textwrap
from ..KineticModel import KineticModel

def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_4, K1, K2 = ks

    return [ (- k1*Ac*E + k_4*An + (k1*K1*Ac*E)/(1+K1) 
                    + (k1*K2*Ac*E - k1*Ac**2*E)/(Ac+K2+K1*Ac+K1*K2) 
                    + (k_4*K2*An - k_4*An*Ac)/(Ac+K2)),
             - k1*Ac*E,
             + k1*Ac*E,
             - k_4*An + (k1*Ac**2*E)/(Ac+K2+K1*Ac+K1*K2) + (k_4*Ac*An)/(Ac+K2)]

model = KineticModel(
    name = "two_int_int_hyd_ss",
    description = textwrap.dedent("""\
        Simple model with distinct intermediates and direct int hydrolysis:

            Ac + E  ---> I1      (k1)
            I1      ---> I2 + U  (k2)
            I1      ---> Ac + U  (k3)
            I2 + Ac <==> An      (k4, k_4)
            I2      ---> Ac      (k5)

            Steady-state approximation with K1 = k3/k2, K2 = k5/k4"""),
    kin_sys = equations,
    ks_guesses = [0.02, 0.1, 10, 10],
    starting_concs_guesses = [50, 50],
    starting_concs_constant = [0, 0],
    parameter_names = ["k1", "k_4", "K1", "K2", "[Acid]0", "[EDC]0"],
    legend_names = ["Acid", "EDC", "Urea", "Anhydride"],
    top_plot = [1, 2],
    bottom_plot = [0, 3],
    sort_order = [1, 3, 2, 0],
    int_eqn = [
        lambda cs, ks: (ks[0]*cs[0]**2*cs[1])/(cs[0]+ks[3]+ks[2]*cs[0]+ks[2]*ks[3]),
        lambda cs, ks: ks[1]*cs[3],
        lambda cs, ks: (ks[1]*cs[0]*cs[3])/(cs[0]+ks[3]),
    ],
    int_eqn_desc = [
        "(k1*Ac^2*E)/(Ac+K2+K1*Ac+K1*K2)",
        "k_4*An",
        "(k_4*Ac*An)/(Ac+K2)",
    ]
    )