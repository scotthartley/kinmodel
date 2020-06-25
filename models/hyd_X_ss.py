import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    An, Ac = concs
    k1, a = ks

    return [-k1*An + k1*Ac*An/(Ac+a),
            k1*An - k1*Ac*An/(Ac+a) + k1*a*An/(Ac+a)]


model = KineticModel(
    name="hyd_X_ss",
    description=textwrap.dedent("""\
        Simple decomposition with exchange:

            An <--> I + Ac  (k1, km1)
             I ---> Ac      (k2)

        Steady-state approximation with a = k2/km1.
        Orders: k1, a; An, Ac.\
        """),
    kin_sys=equations,
    ks_guesses=[0.005, 5],
    ks_constant=[],
    conc0_guesses=[5, 10],
    conc0_constant=[],
    k_var_names=["k1", "a"],
    k_const_names=[],
    conc0_var_names=["[Anh]0", "[Ac]0"],
    conc0_const_names=[],
    legend_names=["Anhydride", "Acid"],
    top_plot=[],
    bottom_plot=[0, 1],
    sort_order=[0, 1],
    int_eqn=[],
    int_eqn_desc=[],
    lifetime_conc=[],
    rectime_conc=[],
    )
