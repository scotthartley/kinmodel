import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    Ac, E, U, L, C = concs
    k1, K, EM, k2L, k2C, k3, k4, kE = ks

    p = Ac/(Ac+L)  # Approx fraction of diacid that is monomer.

    # Return the equations for Ac', E', U', L', C'
    return [(- k1*Ac*E/2 + k2L*L + k2C*C + k1*K*Ac*E/(K+Ac+p*EM)/2
             - k1*Ac**2*E/(K+Ac+p*EM)/2 - k1*p*EM*Ac*E/(K+Ac+p*EM)/2),
            - k1*Ac*E - kE*E,
            + k1*Ac*E + kE*E,
            + k1*Ac**2*E/(K+Ac+p*EM) - k2L*L - k3*(1-p)*Ac + k4*C*Ac,
            + k1*p*EM*Ac*E/(K+Ac+p*EM) - k2C*C + k3*(1-p)*Ac - k4*C*Ac]


model = KineticModel(
    name="DA_simple_Cyc_E_hyd_ss",
    description=textwrap.dedent("""\
        Simple model for diacid assembly with competing macrocyclization
        and polymerization, direct EDC hydrolysis, and cyclization of
        longer diacids.

             DAn + E ---> In         (k1)
                  In ---> DAn + U    (kiAc)
            In + DAn ---> DAn+m + U  (kiL)
                  I0 ---> C + U      (kiC)
               DAn+m ---> DAn + DAm  (k2L)
                   C ---> DA0        (k2C)
                 DAn ---> C + DAn-1  (k3)
             C + DAn ---> DAn+1      (k4)
                   E ---> U          (kE)

        where DAn is a diacid of length n, In is the activated
        intermediate of length n, U is the urea, C is the macrocycle.

        The system above is then translated to the experimentally
        observed concentrations: Ac (total diacid), L (total linear
        anhydride), C (total cyclic anhydride), E (EDCI), and U (urea).

        Steady-state approximation in I: K = kiAc/kiL, EM - kiC/kiL.
        Orders: k1, K, EM, k2L, k2C, k3, k4, kE; Ac, E, U, L, C.\
        """),
    kin_sys=equations,
    ks_guesses=[0.02, 100, 10, 0.5, 1, 0.1, 0.1, 0.1],
    ks_constant=[],
    conc0_guesses=[100, 400],
    conc0_constant=[0, 0, 0],
    k_var_names=["k1", "K", "EM", "k2L", "k2C", "k3", "k4", "kE"],
    k_const_names=[],
    conc0_var_names=["[Acid]0", "[EDC]0"],
    conc0_const_names=["[U]0", "[L]0", "[C]0"],
    legend_names=["Acid", "EDC", "Urea", "Linear", "Cyclic"],
    top_plot=[1, 2],
    bottom_plot=[0, 3, 4],
    sort_order=[2, 3, 4, 0, 1],
    int_eqn=[
        lambda cs, ks: (ks[0]*cs[0]**2*cs[1] /
                        (ks[1]+cs[0]+(cs[0]/(cs[0]+cs[3]))*ks[2])),
        lambda cs, ks: ks[3]*cs[3],
        lambda cs, ks: (ks[0]*(cs[0]/(cs[0]+cs[3]))*ks[2]*cs[0]*cs[1] /
                        (ks[1]+cs[0]+(cs[0]/(cs[0]+cs[3]))*ks[2])),
        lambda cs, ks: ks[4]*cs[4],
        ],
    int_eqn_desc=[
        "k1*Ac^2*E/(K+Ac+p*EM)",
        "k2L*L",
        "k1*p*EM*Ac*E/(K+Ac+p*EM)",
        "k2C*C",
        ],
    lifetime_conc=[3, 4],
    rectime_conc=[0],
    )
