import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k_2, k4, kw, K = ks

    return [(- k1*Ac*E + k_2*An + 2*(kw+k4*Ac)*An - (k1*Ac**2*E)/(Ac+K)
             - (k_2*An*Ac)/(Ac+K) + (k1*K*Ac*E)/(Ac+K) + (k_2*K*An)/(Ac+K)),
            - k1*Ac*E,
            + k1*Ac*E,
            (- k_2*An + (k1*Ac**2*E)/(Ac+K) + (k_2*An*Ac)/(Ac+K)
             - (kw+k4*Ac)*An)]


model = KineticModel(
    name="shared_int_An_hyd_gb_ss",
    description=textwrap.dedent("""\
        Simple model with shared acylpyridinium intermediate, acetate catalysis
        of anhydride hydrolysis:

             Ac + E ---> I + U  (k1)
             I + Ac <==> An     (k2, k-2)
                  I ---> Ac     (k3)
            An + Ac ---> 3Ac    (k4)

            Steady-state approximation with K=k3/k2"""),
    kin_sys=equations,
    ks_guesses=[0.02, 0.03, 0.1, 0.1, 10],
    ks_constant=[],
    conc0_guesses=[50, 50],
    conc0_constant=[0, 0],
    k_var_names=["k1", "k-2", "k4", "kw", "K"],
    k_const_names=[],
    conc0_var_names=["[Acid]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[An]0"],
    legend_names=["Acid", "EDC", "Urea", "Anhydride"],
    top_plot=[1, 2],
    bottom_plot=[0, 3],
    sort_order=[1, 3, 2, 0],
    int_eqn=[],
    int_eqn_desc=[]
    )
