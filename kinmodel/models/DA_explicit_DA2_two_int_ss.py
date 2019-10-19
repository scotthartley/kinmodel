import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    DA1, E, U, DA2, C = concs
    k1, K1, EM1, K2, EM2, k2C, k2L = ks

    # Return the equations for concs
    return [
        (k2L*DA2 - k1*DA1*E - (k2C*C*DA1)/(EM2 + K2 + DA1)
            - (k2L*DA1*DA2)/(EM2 + K2 + DA1) - (k1*DA1**2*E)/(EM1 + K1 + DA1)
            + (K2*k2C*C)/(EM2 + K2 + DA1) + (K2*k2L*DA2)/(EM2 + K2 + DA1)
            + (K1*k1*DA1*E)/(EM1 + K1 + DA1)),
        -k1*DA1*E,
        +k1*DA1*E,
        ((DA1*(k2C*C + k2L*DA2))/(EM2 + K2 + DA1) - k2L*DA2
            + (k1*DA1**2*E)/(EM1 + K1 + DA1)),
        ((EM2*(k2C*C + k2L*DA2))/(EM2 + K2 + DA1) - k2C*C
            + (EM1*k1*DA1*E)/(EM1 + K1 + DA1)),
    ]


model = KineticModel(
    name="DA_explicit_DA2_two_int_ss",
    description=textwrap.dedent("""\
        Simple model for diacid assembly with explicit consideration of
        linear anhydride intermediates, capped at the dimer (DA2).
        Separate intermediates for EDC consumption and anhydride
        exchange are used.

             DA1 + E ---> I1        (k1)
                  I1 ---> DA1 + U   (kih)
                  I1 ---> C + U     (kiC)
            I1 + DA1 ---> DA2       (kiL)
                 DA2 <--> DA1 + Ip1 (k2L, km2L)
                   C <--> Ip1       (k2C, km2C)
                 Ip1 ---> DA1       (k3)

        Steady-state approximations with K1 = kih/kiL, EM1 = kiC/kiL,
        K2 = k3/km2L, and EM2 = km2C/km2L.
        Orders: k1, K1, EM1, K2, EM2, k2C, k2L; DA1, E, U, DA2, C.\
        """),
    kin_sys=equations,
    ks_guesses=[0.01, 25, 10, 0.01, 10, 100, 100],
    ks_constant=[],
    conc0_guesses=[25, 50],
    conc0_constant=[0, 0, 0],
    k_var_names=["k1", "K1", "EM1", "K2", "EM2", "k2C", "k2L"],
    k_const_names=[],
    conc0_var_names=["[DA1]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[DA2]0", "[C]0"],
    legend_names=["DA1", "EDC", "Urea", "DA2", "Cy"],
    top_plot=[1, 2],
    bottom_plot=[0, 3, 4],
    sort_order=[2, 3, 4, 0, 1],
    int_eqn=[
        ],
    int_eqn_desc=[
        ],
    lifetime_conc=[3, 4],
    rectime_conc=[0],
    )
