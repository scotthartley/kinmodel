import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    DA1, E, U, DA2, DA3, C = concs
    k1, K1, EM1, K2, EM2, k2C, k2L = ks

    # Return the equations for concs
    return [
        (DA3*k2L + k2L*DA2 - k1*DA1*E
            - (DA1*(DA3*k2L + k2C*C + k2L*DA2))/(EM2 + K2 + DA1 + DA2)
            + (K2*(DA3*k2L + k2C*C + k2L*DA2))/(EM2 + K2 + DA1 + DA2)
            - (k1*DA1**2*E)/(EM1 + K1 + DA1 + DA2)
            - (DA3*k2L*DA1)/(K2 + DA1) + (K1*k1*DA1*E)/(EM1 + K1 + DA1 + DA2)
            - (k1*DA1*DA2*E)/(K1 + DA1)),
        - k1*DA1*E - k1*DA2*E,
        + k1*DA1*E + k1*DA2*E,
        (DA3*k2L - k2L*DA2 - k1*DA2*E
            + (DA1*(DA3*k2L + k2C*C + k2L*DA2))/(EM2 + K2 + DA1 + DA2)
            - (DA2*(DA3*k2L + k2C*C + k2L*DA2))/(EM2 + K2 + DA1 + DA2)
            + (DA3*K2*k2L)/(K2 + DA1) + (k1*DA1**2*E)/(EM1 + K1 + DA1 + DA2)
            + (K1*k1*DA2*E)/(K1 + DA1)
            - (k1*DA1*DA2*E)/(EM1 + K1 + DA1 + DA2)),
        ((DA2*(DA3*k2L + k2C*C + k2L*DA2))/(EM2 + K2 + DA1 + DA2) - 2*DA3*k2L
            + (DA3*k2L*DA1)/(K2 + DA1)
            + (k1*DA1*DA2*E)/(EM1 + K1 + DA1 + DA2)
            + (k1*DA1*DA2*E)/(K1 + DA1)),
        ((EM2*(DA3*k2L + k2C*C + k2L*DA2))/(EM2 + K2 + DA1 + DA2) - k2C*C
            + (EM1*k1*DA1*E)/(EM1 + K1 + DA1 + DA2)),
    ]


model = KineticModel(
    name="DA_explicit_DA3_two_int_ss",
    description=textwrap.dedent("""\
        Simple model for diacid assembly with explicit consideration of
        linear anhydride intermediates, capped at the trimer (DA3).
        Separate intermediates for EDC consumption and anhydride
        exchange are used.

             DA1 + E ---> I1         (k1)
                  I1 ---> DA1 + U    (kih)
                  I1 ---> C + U      (kiC)
            I1 + DA1 ---> DA2 + U    (kiL)
            I1 + DA2 ---> DA3 + U    (kiL)
                 DA2 <--> DA1 + Ip1  (k2L, km2L)
                   C <--> Ip1        (k2C, km2C)
                 DA3 <--> DA2 + Ip1  (k2L, km2L)
                 DA3 <--> Ip2 + DA1  (k2L, km2L)
                 Ip1 ---> DA1        (k3)
                 Ip2 ---> DA2        (k3)
             DA2 + E ---> I2         (k1)
                  I2 ---> DA2 + U    (kih)
            I2 + DA1 ---> DA3 + U    (kiL)

        Steady-state approximations with K1 = kih/kiL, EM1 = kiC/kiL,
        K2 = k3/km2L, and EM2 = km2C/km2L.
        Orders: k1, K1, EM1, K2, EM2, k2C, k2L; DA1, E, U, DA2, DA3, C.\
        """),
    kin_sys=equations,
    ks_guesses=[0.01, 25, 10, 0.01, 10, 100, 100],
    ks_constant=[],
    conc0_guesses=[25, 50],
    conc0_constant=[0, 0, 0, 0],
    k_var_names=["k1", "K1", "EM1", "K2", "EM2", "k2C", "k2L"],
    k_const_names=[],
    conc0_var_names=["[DA1]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[DA2]0", "[DA3]0", "[C]0"],
    legend_names=["DA1", "EDC", "Urea", "DA2", "DA3", "Cy"],
    top_plot=[1, 2],
    bottom_plot=[0, 3, 4, 5],
    sort_order=[],
    int_eqn=[
        lambda cs, ks: ((ks[6]*ks[4]*cs[4]) /
                        (ks[4]+ks[3]+cs[0]+cs[3])),
        lambda cs, ks: ((ks[5]*ks[4]*cs[5]) /
                        (ks[4]+ks[3]+cs[0]+cs[3])),
        lambda cs, ks: ((ks[6]*ks[4]*cs[3]) /
                        (ks[4]+ks[3]+cs[0]+cs[3])),
        lambda cs, ks: ks[5]*cs[5],
        lambda cs, ks: ((ks[2]*ks[0]*cs[0]*cs[1]) /
                        (ks[2]+ks[1]+cs[0]+cs[3])),
        ],
    int_eqn_desc=[
        "(k2L*EM2*DA3)/(EM2+K2+DA1+DA2)",
        "(k2C*EM2*C)/(EM2+K2+DA1+DA2)",
        "(k2L*EM2*DA2)/(EM2+K2+DA1+DA2)",
        "k2C*C",
        "(EM1*k1*DA1*E)/(EM1+K1+DA1+DA2)"
        ],
    calcs=[
        lambda cs, ts, ks, ints: ints[4][1],
        lambda cs, ts, ks, ints: ints[4][1] / cs[:, 1][0],
        lambda cs, ts, ks, ints: ints[3][1],
        lambda cs, ts, ks, ints: ints[2][1],
        lambda cs, ts, ks, ints: ints[0][1],
        lambda cs, ts, ks, ints: ints[1][1],
        ],
    calcs_desc=[
        "C produced directly from EDC ∫(EM1*k1*DA1*E)/(EM1+K1+DA1+DA2)dt",
        "C yield directly from EDC ∫(EM1*k1*DA1*E)/(EM1+K1+DA1+DA2)dt/E0",
        "Total C hydrolysis ∫(k2C*C)dt",
        "C produced from DA2 exchange ∫(k2L*EM2*DA2)/(EM2+K2+DA1+DA2)dt",
        "C produced from DA3 exchange ∫(k2L*EM2*DA3)/(EM2+K2+DA1+DA2)dt",
        "C produced from C after decomp ∫(k2C*EM2*C)/(EM2+K2+DA1+DA2)dt",
        ],
    lifetime_conc=[3, 4, 5],
    rectime_conc=[0],
    )
