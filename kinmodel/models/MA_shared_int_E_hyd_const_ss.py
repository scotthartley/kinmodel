import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_2, K, k4 = ks

    return [(- k1*Ac*E - (k1*Ac**2*E)/(Ac+K) - (k_2*An*Ac)/(Ac+K) + k_2*An
             + (k1*K*Ac*E)/(Ac+K) + (k_2*K*An)/(Ac+K)),
            - k1*Ac*E - k4*E,
            + k1*Ac*E + k4*E,
            - k_2*An + (k1*Ac**2*E)/(Ac+K) + (k_2*An*Ac)/(Ac+K)]


model = KineticModel(
    name="MA_shared_int_E_hyd_const_ss",
    description=textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate and direct
        EDC hydrolysis:

            Ac + E ---> I + U  (k1)
            I + Ac <==> An     (k2, k-2)
                 I ---> Ac     (k3)
                 E ---> U      (k4)

        Steady-state approximation with K=k3/k2. k4 is fixed.
        Orders: k1, k_2, K, k4; Ac, E, U, An.\
        """),
    kin_sys=equations,
    ks_guesses=[0.02, 0.03, 10],
    ks_constant=[0.01],
    conc0_guesses=[50, 50],
    conc0_constant=[0, 0],
    k_var_names=["k1", "k-2", "K"],
    k_const_names=["k4"],
    conc0_var_names=["[Acid]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[An]0"],
    legend_names=["Acid", "EDC", "Urea", "Anhydride"],
    top_plot=[1, 2],
    bottom_plot=[0, 3],
    sort_order=[1, 3, 2, 0],
    int_eqn=[
            lambda cs, ks: ks[1]*cs[3],
            lambda cs, ks: (ks[0]*cs[0]**2*cs[1])/(cs[0]+ks[2]),
            lambda cs, ks: (ks[1]*cs[3]*cs[0])/(cs[0]+ks[2]),
            lambda cs, ks: ks[0]*cs[1]*cs[0],
            lambda cs, ks: ks[3]*cs[1],
            lambda cs, ks: (ks[0]*ks[2]*cs[0]*cs[1])/(cs[0]+ks[2]),
            lambda cs, ks: (ks[1]*ks[2]*cs[3])/(cs[0]+ks[2]),
            ],
    int_eqn_desc=[
            "k_2*An",
            "(k1*Ac^2*E)/(Ac+K)",
            "(k_2*An*Ac)/(Ac+K)",
            "k1*E*Ac",
            "k4*E",
            "(k1*K*Ac*E)/(Ac+K)",
            "(k_2*K*An)/(Ac+K)",
            ],
    calcs=[
            lambda cs, ts, ks, ints: max(cs[:, 3]),
            lambda cs, ts, ks, ints: (
                    ts[np.argsort(cs[:, 3])[-1]]),
            lambda cs, ts, ks, ints: cs[:, 3][-1],
            lambda cs, ts, ks, ints: ints[1][1]/cs[:, 1][0],
            lambda cs, ts, ks, ints: (ints[0][1] - ints[2][1])/cs[:, 1][0],
            lambda cs, ts, ks, ints: (ints[3][1] - ints[5][1])/cs[:, 1][0],
            lambda cs, ts, ks, ints: (ints[4][1] + ints[5][1])/cs[:, 1][0],
            lambda cs, ts, ks, ints: ints[2][1]/ints[6][1],
            ],
    calcs_desc=[
            "Maximum An",
            "t to max An",
            "Final An",
            "Yield (∫(k1*Ac^2*E)/(Ac+K)dt)/E0",
            "Yield (∫(k_2An)dt - ∫(k_2*An*Ac)/(Ac+K)dt)/E0",
            "Yield (∫(k1*E*Ac)dt - ∫(k1*K*Ac*E)/(Ac+K)dt)/E0",
            "Wasted E (∫(k4*E)dt + ∫(k1*K*Ac*E)/(Ac+K)dt)/E0",
            "Net exchange ∫(k_2*An*Ac)/(Ac+K)dt / ∫((k_2*K*An)/(Ac+K))dt",
            ],
    lifetime_conc=[3],
    rectime_conc=[0],
    )
