import textwrap
from ..KineticModel import KineticModel


def equations(concs, t, *ks):
    E, U = concs
    k, = ks

    return [-k*E,
            +k*E]


model = KineticModel(
    name="first_ord",
    description=textwrap.dedent("""\
        Simple first order decay

            E ---> U       (k)\
        """),
    kin_sys=equations,
    ks_guesses=[1],
    ks_constant=[],
    conc0_guesses=[100],
    conc0_constant=[0],
    k_var_names=["k"],
    k_const_names=[],
    conc0_var_names=["[EDC]0"],
    conc0_const_names=["[U]0"],
    legend_names=["EDC", "Urea"],
    top_plot=[],
    bottom_plot=[0, 1],
    sort_order=[0, 1],
    int_eqn=[],
    int_eqn_desc=[],
    calcs=[
            lambda cs, ks, ints: max(cs[:, 0]),
            lambda cs, ks, ints: cs[:, 0][-1],],
    calcs_desc=[
            "Maximum E",
            "Final E", ],
    )
