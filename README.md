# kinmodel: A python package for chemical kinetic models

kinmodel is a python package for modeling chemical kinetics, and in particular for fitting kinetic models to experimental concentration vs time data. It defines the KineticModel class, which contains a system of differential equations describing a system of chemical reactions and methods for simulating the system and fitting it to experimental concentration vs time data. It is a fairly straightforward implementation of methods in Scipy.

*Note that elements of the model definition are executed within the program using Python's exec() or eval functions! This is a security risk if models are loaded from untrusted sources.*

## Requirements

- Python 3.6
- Numpy
- Scipy 1.2.1
- Matplotlib
- joblib
- tqdm
- PyYAML

## Installation

The package can be installed as usual with pip.

## Use

At this point, primary interaction with kinmodel will probably be through the fit_kinetics.py executable. Basic usage is of the form `fit_kinetics model_name exp_data.csv`, where `model_name` defines the kinetic model to be applied and `exp_data.csv` is a csv file containing experimental concentration vs time data for relevant species. The csv file can contain multiple experiments that will be fitted simultaneously for all rate constants. Experiments are separated by title lines that must contain a title in column 1 that cannot be interpreted as a number followed by empty cells for the remaining columns.

The program will output txt files of the fit parameters and pdf files of plots. At present, it produces up to two subplots so different species can be visualized separately. There are a number of command line options. Help is available via `fit_kinetics -h`.

Data can also be simulated given a set of parameters using the model_kinetics.py executable. The format is `model_kinetics model_name simulation_time par1 par2 par3 ...`. Default starting concentrations associated with the model can be overridden by including them at the end of the parameters list. Options can be specified to control the output. Ranges for the parameters can be set along with an optional target number of simulations (`-n`): in this case, the ranges will be divided into equal segments such that the total number of simulations is at most the target. More information is available from `model_kinetics -h`.

Some models are included as part of the default installation, but new ones can be defined in a simple YAML format. The models should be loaded into either the working directory, a `models` subdirectory of the working directory, or in the user's data directory for the program, and must have the `.yaml` file extension. They should be the only files in these locations with that extension. The user data directory for the program can be identified by running `fit_kinetics -h` (the model directories are listed right after the descriptions of different options). A simple first order decay is included as a default model:

```
name: first_ord
description: |
    Simple first order decay

        S ---> P       (k)

    Orders: k; S, P.
eq_function: |
    def equations(concs, t, *ks):
        S, P = concs
        k, = ks

        return [-k*S,
                +k*S,
        ]
k_var:
    - name: k
      guess: 1
conc0_var:
    - name: "[S]0"
      guess: 100
conc0_const:
    - name: "[P]0"
      value: 0
species:
    - name: Starting material
      plot: bottom
      sort: 0
    - name: Product
      plot: bottom
      sort: 1
calcs:
    - desc: "Maximum S"
      func: "max(c[0])"
    - desc: "Final S"
      func: "c[0][-1]"
```

Key fields are `eq_function`, which is a Python function of concentrations (concs), time (t), and parameters (ks) that returns the results of the kinetic equations (first derivatives of concentrations) defining the system. The `sort` field for each `species` is used to relate to the order of columns in the input.

Models can also be defined with the IndirectKineticModel class. This allows normal KineticModel mechanisms to be used in cases where the experimental observables are a function of the underlying concentrations (e.g., oligomers where the total functional group concentration is known but individual concentrations are not). These are defined as in:

```
name: "DA_explicit_DA3_two_int_ss_ind"
type: indirect
parent_model_name: "DA_explicit_DA3_two_int_ss"
description: |
    Indirect version of the DA_explicit_DA3_two_int_ss model, using total
    diacid and total anhydride concentration.
species:
    - name: "Diacid"
      plot: bottom
      sort: 2
      map: c[0] + c[3] + c[4]
    - name: "EDC"
      plot: top
      sort: 3
      map: c[1]
    - name: "Urea"
      plot: top
      sort: 4
      map: c[2]
    - name: "Linear"
      plot: bottom
      sort: 0
      map: c[3] + 2*c[4]
    - name: "Cyclic"
      plot: bottom
      sort: 1
      map: c[5]
```

The key here is the `map` field for each species, which relate them to concentration in the underlying model.

This indirect model uses the following as its underlying mechanism:

```
name: DA_explicit_DA3_two_int_ss
description: |
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
    Orders: k1, K1, EM1, K2, EM2, k2C, k2L; DA1, E, U, DA2, DA3, C.
eq_function: |
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
k_var: 
    - name: "k1"
      guess: 1
    - name: "K1"
      guess: 40
    - name: "EM1"
      guess: 50
    - name: "K2"
      guess: 45
    - name: "EM2"
      guess: 100
    - name: "k2C"
      guess: 1e-2
    - name: "k2L"
      guess: 2e-2
k_const:
conc0_var:
    - name: "[DA1]0"
      guess: 25
    - name: "[EDC]0"
      guess: 50
conc0_const:
    - name: "[U]0"
      value: 0
    - name: "[DA2]0"
      value: 0
    - name: "[DA3]0"
      value: 0
    - name: "[C]0"
      value: 0
species:
    - name: "DA1"
      plot: bottom
    - name: "EDC"
      plot: top
    - name: "Urea"
      plot: top
    - name: "DA2"
      plot: bottom
    - name: "DA3"
      plot: bottom
    - name: "Cy"
      plot: bottom
integrals:
    - desc: "(k2L*EM2*DA3)/(EM2+K2+DA1+DA2)"
      func: "((k[6]*k[4]*c[4]) / (k[4]+k[3]+c[0]+c[3]))"
    - desc: "(k2C*EM2*C)/(EM2+K2+DA1+DA2)"
      func: "((k[5]*k[4]*c[5]) / (k[4]+k[3]+c[0]+c[3]))"
    - desc: "(k2L*EM2*DA2)/(EM2+K2+DA1+DA2)"
      func: "((k[6]*k[4]*c[3]) / (k[4]+k[3]+c[0]+c[3]))"
    - desc: "k2C*C"
      func: "k[5]*c[5]"
    - desc: "(EM1*k1*DA1*E)/(EM1+K1+DA1+DA2)"
      func: "((k[2]*k[0]*c[0]*c[1]) / (k[2]+k[1]+c[0]+c[3]))"
calcs:
    - desc: "C produced directly from EDC ∫(EM1*k1*DA1*E)/(EM1+K1+DA1+DA2)dt"
      func: "i[4]"
    - desc: "C yield directly from EDC ∫(EM1*k1*DA1*E)/(EM1+K1+DA1+DA2)dt/E0"
      func: "i[4] / c[1][0]"
    - desc: "Total C hydrolysis ∫(k2C*C)dt"
      func: "i[3]"
    - desc: "C produced from DA2 exchange ∫(k2L*EM2*DA2)/(EM2+K2+DA1+DA2)dt"
      func: "i[2]"
    - desc: "C produced from DA3 exchange ∫(k2L*EM2*DA3)/(EM2+K2+DA1+DA2)dt"
      func: "i[0]"
    - desc: "C produced from C after decomp ∫(k2C*EM2*C)/(EM2+K2+DA1+DA2)dt"
      func: "i[1]"
lifetime_concs:
    - 3
    - 4
    - 5
rectime_concs:
    - 0
```

A useful feature of kinmodel is that the KineticModel objects can contain calculations that will be performed on the results of the regression. Various quantities, like maximum concentrations, can be calculated. There are two aspects of this. The `integrals` field defines equations that will be integrated across the concentration vs time data; `desc` is used to describe them in the output and `func` is the function that will be integrated of parameters (k) and concentrations (c), indexed in the order they are listed elsewhere (starting from 0). `calcs` are functions of concentration (c), time (t), parameters (k), and integrals (i) that are calculated at the end for a given run.

