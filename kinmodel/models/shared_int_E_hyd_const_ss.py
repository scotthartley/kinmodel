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
    name="shared_int_E_hyd_const_ss",
    description=textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate and direct
        EDC hydrolysis:

            Ac + E ---> I + U  (k1)
            I + Ac <==> An     (k2, k-2)
                 I ---> Ac     (k3)
                 E ---> U      (k4)

            Steady-state approximation with K=k3/k2
            k4 is fixed"""),
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
            lambda cs, ks: (ks[0]*cs[0]**2*cs[1])/(cs[0]+ks[3]),
            lambda cs, ks: ks[1]*cs[3],
            lambda cs, ks: (ks[1]*cs[3]*cs[0])/(cs[0]+ks[3]),
            lambda cs, ks: ks[0]*cs[1]*cs[0],
            lambda cs, ks: ks[2]*cs[1], ],
    int_eqn_desc=[
            "(k1*Ac^2*E)/(Ac+K)",
            "k_2*An",
            "(k_2*An*Ac)/(Ac+K)",
            "k1*E*Ac",
            "k4*E", ]
    )
