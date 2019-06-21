import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_2, K, k4, kE = ks

    return [(- k1*Ac*E - (k1*Ac**2*E)/(Ac+K) - (k_2*An*Ac)/(Ac+K) + k_2*An
             + (k1*K*Ac*E)/(Ac+K) + (k_2*K*An)/(Ac+K)),
            - k1*Ac*E - k4*E + kE,
            + k1*Ac*E + k4*E,
            - k_2*An + (k1*Ac**2*E)/(Ac+K) + (k_2*An*Ac)/(Ac+K)]


model = KineticModel(
    name="MA_shared_int_E_hyd_ss_AddE",
    description=textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate and direct
        EDC hydrolysis. This model is intended primarily for simulation.

            Ac + E ---> I + U  (k1)
            I + Ac <==> An     (k2, k-2)
                 I ---> Ac     (k3)
                 E ---> U      (k4)

            E added at rate kE

        Steady-state approximation with K=k3/k2.
        Orders: k1, k_2, K, k4, kE; Ac, E, U, An.\
        """),
    kin_sys=equations,
    ks_guesses=[0.02, 0.03, 10, 0.1],
    ks_constant=[1],
    conc0_guesses=[50, 50],
    conc0_constant=[0, 0],
    k_var_names=["k1", "k-2", "K", "k4"],
    k_const_names=["kE"],
    conc0_var_names=["[Acid]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[An]0"],
    legend_names=["Acid", "EDC", "Urea", "Anhydride"],
    top_plot=[1, 2],
    bottom_plot=[0, 3],
    sort_order=[1, 3, 2, 0],
    int_eqn=[
            ],
    int_eqn_desc=[
            ],
    calcs=[
            lambda cs, ks, ints: max(cs[:, 3]),
            lambda cs, ks, ints: cs[:, 3][-1],
            lambda cs, ks, ints: (
                    (cs[:, 3][-1] - cs[:, 3][round(len(cs[:, 3])*0.75)])
                    / cs[:, 3][-1]),
            lambda cs, ks, ints: (
                ((ks[0]*cs[:, 0][-1]**2*cs[:, 1][-1])/(cs[:, 0][-1]+ks[2]))
                / (ks[0]*cs[:, 0][-1]*cs[:, 1][-1] + ks[3]*cs[:, 1][-1])),
            lambda cs, ks, ints: (
                ((ks[0]*cs[:, 0][round(len(cs[:, 0])*0.75)]**2*cs[:, 1][round(len(cs[:, 0])*0.75)])/(cs[:, 0][round(len(cs[:, 0])*0.75)]+ks[2]))
                / (ks[0]*cs[:, 0][round(len(cs[:, 0])*0.75)]*cs[:, 1][round(len(cs[:, 0])*0.75)] + ks[3]*cs[:, 1][round(len(cs[:, 3])*0.75)])),
            ],
    calcs_desc=[
            "Maximum An",
            "Final An",
            "(An_final-An_75%)/An_final",
            "Final efficiency (An produced/E consumed)",
            "Efficiency at 75% t"
            ],
    lifetime_conc=[],
    rectime_conc=[],
    )
