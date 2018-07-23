import textwrap
from ..KineticModel import KineticModel

def simple_monoacid(concs, t, *ks):
    Ac, E, U, An = concs
    k1, K, k2 = ks

    return [ - k1*E*Ac + 2*k2*An + (k1*K*Ac*E)/(K+Ac) - (k1*E*Ac**2)/(K+Ac),
             - k1*E*Ac,
             + k1*E*Ac, 
             + (k1*E*Ac**2)/(K+Ac) - k2*An ]

model = KineticModel(
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
        "k2*An",
    ]
    )