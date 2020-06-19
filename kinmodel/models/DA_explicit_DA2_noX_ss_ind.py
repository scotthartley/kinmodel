import textwrap
import numpy as np
from ..KineticModel import IndirectKineticModel


model = IndirectKineticModel(
    name="DA_explicit_DA2_noX_ss_ind",
    parent_model_name="DA_explicit_DA2_noX_ss",
    description=textwrap.dedent("""\
        Indirect version of the DA_explicit_DA2_noX_ss model, using total
        diacid and total anhydride concentration.\
        """),
    conc_mapping=lambda c: np.array([c[:, 0]+c[:, 3],
                                     c[:, 1],
                                     c[:, 2],
                                     c[:, 3],
                                     c[:, 4]]).transpose(),
    legend_names=["Diacid", "EDC", "Urea", "Linear", "Cyclic"],
    top_plot=[1, 2],
    bottom_plot=[0, 3, 4],
    sort_order=[2, 3, 4, 0, 1],
    )
