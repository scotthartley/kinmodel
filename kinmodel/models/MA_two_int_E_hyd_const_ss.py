import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    Ac, E, U, An = concs
    k1, k4, K1, K2, kh = ks

    return [(- k1*Ac*E - (k1*Ac**2*E)/(Ac+K1) + (k1*K1*Ac*E)/(Ac+K1)
             + k4*An - (k4*Ac*An)/(Ac+K2) + (k4*K2*An)/(Ac+K2)),
            - k1*Ac*E - kh*E,
            + k1*Ac*E + kh*E,
            + (k1*Ac**2*E)/(Ac+K1) - k4*An + (k4*Ac*An)/(Ac+K2)]


model = KineticModel(
    name="MA_two_int_E_hyd_const_ss",
    description=textwrap.dedent("""\
        Simple model with distinct intermediates and direct E hydrolysis:

             Ac + E ---> I1       (k1)
            I1 + Ac ---> An + U   (k2)
                 I1 ---> Ac + U   (k3)
                 An <==> I2 + Ac  (k4, k_4)
                 I2 ---> Ac       (k5)
                  E ---> U        (kh)

        Steady-state approximation with K1=k3/k2, K2=k5/k_4. kh is fixed.
        Orders: k1, k4, K1, K2, kh; Ac, E, U, An.\
        """),
    kin_sys=equations,
    ks_guesses=[0.02, 200, 10, 0.001],
    ks_constant=[1e-3],
    conc0_guesses=[50, 50],
    conc0_constant=[0, 0],
    k_var_names=["k1", "k4", "K1", "K2"],
    k_const_names=["kh"],
    conc0_var_names=["[Acid]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[An]0"],
    legend_names=["Acid", "EDC", "Urea", "Anhydride"],
    top_plot=[1, 2],
    bottom_plot=[0, 3],
    sort_order=[1, 3, 2, 0],
    int_eqn=[],
    int_eqn_desc=[],
    lifetime_conc=[3],
    rectime_conc=[0],
    )
