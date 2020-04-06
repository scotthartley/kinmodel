import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    DA1, E, U, DA2, DA3, DA4, C = concs
    k1, EM1, EM2, k2C, k2L, K1, K2 = ks

    # Return the equations for concs
    return [
        (k2L*DA2 + k2L*DA3 + k2L*DA4
            + (K2*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            - k1*DA1*E
            - (DA1*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            - (k1*DA1**2*E)/(EM1 + K1 + DA1 + DA2 + DA3)
            - (k2L*DA1*(DA3 + DA4))/(K2 + DA1 + DA2)
            - (k2L*DA1*DA4)/(K2 + DA1) - (k1*DA1*DA2*E)/(K1 + DA1 + DA2)
            - (k1*DA1*DA3*E)/(K1 + DA1)
            + (K1*k1*DA1*E)/(EM1 + K1 + DA1 + DA2 + DA3)),
        - k1*DA1*E - k1*DA2*E - k1*DA3*E,
        + k1*DA1*E + k1*DA2*E + k1*DA3*E,
        (k2L*DA3 - k2L*DA2 + k2L*DA4 - k1*DA2*E
            + (DA1*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            - (DA2*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            + (k1*DA1**2*E)/(EM1 + K1 + DA1 + DA2 + DA3) - (k1*DA2**2*E)/(K1 + DA1 + DA2)
            + (K2*k2L*(DA3 + DA4))/(K2 + DA1 + DA2)
            - (k2L*DA2*(DA3 + DA4))/(K2 + DA1 + DA2)
            + (K1*k1*DA2*E)/(K1 + DA1 + DA2)
            - (k1*DA1*DA2*E)/(EM1 + K1 + DA1 + DA2 + DA3)),
        (k2L*DA4 - 2*k2L*DA3 - k1*DA3*E
            + (DA2*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            - (DA3*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            + (K2*k2L*DA4)/(K2 + DA1)
            + (k2L*DA1*(DA3 + DA4))/(K2 + DA1 + DA2) + (K1*k1*DA3*E)/(K1 + DA1)
            + (k1*DA1*DA2*E)/(EM1 + K1 + DA1 + DA2 + DA3)
            - (k1*DA1*DA3*E)/(EM1 + K1 + DA1 + DA2 + DA3)
            + (k1*DA1*DA2*E)/(K1 + DA1 + DA2)),
        ((DA3*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            - 3*k2L*DA4 + (k1*DA2**2*E)/(K1 + DA1 + DA2)
            + (k2L*DA2*(DA3 + DA4))/(K2 + DA1 + DA2)
            + (k2L*DA1*DA4)/(K2 + DA1)
            + (k1*DA1*DA3*E)/(EM1 + K1 + DA1 + DA2 + DA3)
            + (k1*DA1*DA3*E)/(K1 + DA1)),
        ((EM2*(C*k2C + k2L*DA2 + k2L*DA3 + k2L*DA4))/(EM2 + K2 + DA1 + DA2 + DA3)
            - C*k2C + (EM1*k1*DA1*E)/(EM1 + K1 + DA1 + DA2 + DA3)),
    ]


model = KineticModel(
    name="DA_explicit_DA4_two_int_K_const_ss",
    description=textwrap.dedent("""\
        Simple model for diacid assembly with explicit consideration of
        linear anhydride intermediates, capped at the tetramer (DA4).
        Separate intermediates for EDC consumption and anhydride
        exchange are used. K1 and K2 are held constant.

               DA(n) + E ---> I(n)          (k1)
                    I(n) ---> DA(n) + U     (kih)
                    I(1) ---> C + U         (kiC)
            I(n) + DA(m) ---> DA(n+m) + U   (kiL)
                       C <--> I(1)          (k2C, km2C)
                 DA(n+m) <--> DA(n) + Ip(m) (k2L, km2L)
                   Ip(n) ---> DA(n)         (k3)

        where the numbers in parentheses index the number of repeat
        units in each species. Steady-state approximations with K1 =
        kih/kiL, EM1 = kiC/kiL, K2 = k3/km2L, and EM2 = km2C/km2L.
        Orders: k1, EM1, EM2, k2C, k2L, K1, K2; DA1, E, U, DA2, DA3,
        DA4, C.\
        """),
    kin_sys=equations,
    ks_guesses=[0.01, 10, 10, 100, 100],
    ks_constant=[25, 10],
    conc0_guesses=[25, 50],
    conc0_constant=[0, 0, 0, 0, 0],
    k_var_names=["k1", "EM1", "EM2", "k2C", "k2L"],
    k_const_names=["K1", "K2"],
    conc0_var_names=["[DA1]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[DA2]0", "[DA3]0", "[DA4]0", "[C]0"],
    legend_names=["DA1", "EDC", "Urea", "DA2", "DA3", "DA4", "Cy"],
    top_plot=[1, 2],
    bottom_plot=[0, 3, 4, 5, 6],
    sort_order=[],
    int_eqn=[
        ],
    int_eqn_desc=[
        ],
    lifetime_conc=[3, 4, 5, 6],
    rectime_conc=[0],
    )
