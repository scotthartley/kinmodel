import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    DA1, E, U, DA2, C = concs
    k1, K, EM, k2L, k2C, kXLC, kXC, kEH = ks

    I1d = K+EM+DA1

    # Return the equations for concs
    return [(- k1*DA1*E + (k1*K*DA1*E)/I1d - (k1*DA1**2*E)/I1d
             + kXLC*DA2 + 2*k2L*DA2 + k2C*C - kXC*C*DA1),
            - k1*DA1*E - kEH*E,
            + k1*DA1*E + kEH*E,
            + (k1*DA1**2*E)/I1d - kXLC*DA2 - k2L*DA2 + kXC*C*DA1,
            + (k1*EM*DA1*E)/I1d + kXLC*DA2 - k2C*C - kXC*C*DA1]


model = KineticModel(
    name="DA_explicit_DA2_ss",
    description=textwrap.dedent("""\
        Simple model for diacid assembly with explicit consideration of
        linear anhydride intermediates, capped at the dimer (DA2).
        Direct EDC hydrolysis is included.

            DA1 + E ---> I1 + U   (k1)
                 I1 ---> DA1      (kiH)
                 I1 ---> C        (kiC)
           I1 + DA1 ---> DA2      (kiL)
                DA2 ---> C + DA1  (kXLC)
                DA2 ---> 2*DA1    (k2L)
                  C ---> DA1      (k2C)
           Cy + DA1 ---> DA2      (kXLC)
                  E ---> U        (kEH)

        Steady-state approximation with K = kiH/kiL and EM=kiC/kiL.\
        """),
    kin_sys=equations,
    ks_guesses=[0.02, 100, 50, 1, 1, 0, 0, 0],
    ks_constant=[],
    conc0_guesses=[100, 400],
    conc0_constant=[0, 0, 0],
    k_var_names=["k1", "K", "EM", "k2L", "k2C", "kXLC", "kXC", "kEH"],
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
