import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    DA, E, U, An = concs
    k1, K1, K2, k2 = ks

    return [- k1*E*DA + k2*An + (k1*K1*DA*E)/(1+K1),
            - k1*E*DA - (K2*k1*DA*E**2)/(1+K1),
            + k1*E*DA + (K2*k1*DA*E**2)/(1+K1),
            + (k1*E*DA)/(1+K1) - k2*An]


model = KineticModel(
    name="DA_simple_nopoly_I2_ss",
    description=textwrap.dedent("""\
        Simplest diacid kinetic model with no polymerization, including
        formation of bis(O-acylisourea) intermediate:

            DA + E ---> I1      (k1)
                I1 ---> An + U  (kiAn)
                I1 ---> DA + U  (kiDA)
            I1 + E ---> I2      (kiE)
                I2 ---> I1 + U  (ki2Ac)
                An ---> DA      (k2)

        Steady-state approximation with K1 = kiDA/kiAn, K2 = kiE/kiAn.
        Orders: k1, K1, K2, k2; DA, E, U, An.\
        """),
    kin_sys=equations,
    ks_guesses=[0.02, 1, 1, 0.5],
    ks_constant=[],
    conc0_guesses=[25, 100],
    conc0_constant=[0, 0],
    k_var_names=["k1", "K1", "K2", "k2"],
    k_const_names=[],
    conc0_var_names=["[Acid]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[An]0"],
    legend_names=["Acid", "EDC", "Urea", "Anhydride"],
    top_plot=[1, 2],
    bottom_plot=[0, 3],
    sort_order=[1, 3, 2, 0],
    int_eqn=[
            lambda cs, ks: ks[2]*cs[3],
            lambda cs, ks: ks[0]*cs[0]*cs[1],
            lambda cs, ks: ks[0]*ks[1]*cs[0]*cs[1]/(1+ks[1]),
            ],
    int_eqn_desc=[
            "k2*An",
            "k1*DA*E",
            "k1*K*DA*E/(1+K)",
            ],
    calcs=[
            lambda cs, ts, ks, ints: max(cs[:, 3]),
            lambda cs, ts, ks, ints: cs[:, 3][-1],
            lambda cs, ts, ks, ints: ints[0][1]/cs[:, 1][0],
            lambda cs, ts, ks, ints: (ints[1][1]-ints[2][1])/cs[:, 1][0],
            ],
    calcs_desc=[
            "Maximum An",
            "Final An",
            "An yield from (∫(k2*An)dt)/E0",
            "An yield from (∫(k1*DA*E) - (k1*K*DA*E)/(1+K)dt)/E0",
            ],
    lifetime_conc=[3],
    rectime_conc=[0],
    )
