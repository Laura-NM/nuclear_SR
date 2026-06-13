# LAURA NAVARRO MARTÍNEZ 
# 07/04/2026
#
# PROGRAM: trainA.py

# Approach A:
# This program loades a SEMF per nuclei synthetic dataset, whichs is cut N,Z>=8 for the train set. 
# Known Semiempirical Mass Formula terms are pre-computed and 
# build into a new dataset, the X train dataset.
# The regressor must find the structure that fits this pre-computed terms.
# Run >> python trainA.py << in terminal to do so.

# It saves the model output for analysis.
# To use the model load it as: 
# pysr_model=PySRRegressor.from_file(r"C:\Users\Lauri\Desktop\TFG-code\results\models\trainA\{}")
#                                                                                     -> the model folder.

# Imports. They take time ~ 40 s

import os
os.environ["JULIA_NUM_THREADS"] = "12"  # Intel Core Ultra 7 has ~12 P+E cores
os.environ["JULIA_EXCLUSIVE"] = "0"

import pysr
from pysr import PySRRegressor
import numpy as np
import sympy as sp
import pandas as pd

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. Data port -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

def parse_ame2020(filepath: str) -> pd.DataFrame:
    """Parse the AME2020 mass table from a fixed-width text file.

    Extrapolated values (marked with '#') and uncalculable entries ('*')
    are silently skipped.

    Args:
        filepath (str): Path to the AME2020 table file (mass_1.mas20.txt).

    Returns:
        pd.DataFrame with columns:
            A              — mass number
            Z              — atomic number
            BE_per_A       — binding energy per nucleon (keV)
            Error_per_A    — experimental uncertainty on BE/A (keV)
            Mass_Excess    — atomic mass excess (keV)
            ME_Error       — experimental uncertainty on mass excess (keV)
            Element        — chemical symbol
    """
    data = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    start_parsing = False
    for line in lines:
        if "0  1    1    0    1" in line:
            start_parsing = True
        if not start_parsing or len(line) < 80:
            continue

        try:
            A_str            = line[14:19]
            Z_str            = line[9:14]
            BE_per_A_str     = line[54:67]
            BE_per_A_err_str = line[68:78]
            ME_str           = line[29:43]
            ME_err_str       = line[43:55]
            element_symbol   = line[20:22].strip()

            # Skip extrapolated or missing entries
            if "#" in BE_per_A_str or "#" in BE_per_A_err_str or "*" in BE_per_A_str:
                continue
            if "#" in ME_str or "*" in ME_str:
                continue

            A            = int(A_str)
            Z            = int(Z_str)
            BE_per_A     = float(BE_per_A_str)
            BE_per_A_err = float(BE_per_A_err_str)
            ME           = float(ME_str)
            ME_err       = float(ME_err_str)

            data.append([A, Z, BE_per_A, BE_per_A_err, ME, ME_err, element_symbol])
        except ValueError:
            continue

    return pd.DataFrame(data, columns=[
        "A", "Z", "BE_per_A", "Error_per_A", "Mass_Excess", "ME_Error", "Element",
    ])

raw_df=parse_ame2020(r"C:\Users\Lauri\Desktop\TFG-code\data\mass_1.mas20.txt")

# Exclude very light nuclei (N,Z < 8):
df=raw_df[((raw_df["A"]-raw_df["Z"])>=8)&(raw_df["Z"]>=8)]

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. Variables. [A,Z,N, ...] -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. 
# Include pre-computed variables for an easier find. [A^1/3 , Z(Z-1), (N-Z)^2, pairing delta] 

A=df["A"].values.astype(np.float32)
Z=df["Z"].values.astype(np.float32) # also np.array(df["Z"]).astype(np.float32)
N=A-Z
A_3root=A**(-1/3)
Z_cou=Z**2 / A**(4/3)
NZ_Asim=(A-2*Z)**2 / A**2

def pairing_term(Z,N):
    "(Z,N) arrays must be integer type when passed to pairing term."
    delta=np.zeros(len(Z)) # Row of zeros.
    even_even=(Z%2==0)&(N%2==0)
    odd_odd=(Z%2==1)&(N%2==1)
    delta[even_even]=1.0
    delta[odd_odd]=-1.0

    return delta.astype(np.float32)

delta=pairing_term(Z.astype(int),N.astype(int))
pairing=delta/A**(3/2)

X=pd.DataFrame({
    "A":A,
    "Z":Z,
    "A_3root":A_3root,
    "Z_cou":Z_cou,
    "NZ_Asim":NZ_Asim,
    "pairing":pairing
})

# Real Binding Energy data from AME2020.

BE_per_A=df["BE_per_A"].values.astype(np.float32) # keV / nucleon

# Synthetic Binding Energy from Semiempirical Mass Formula with our coefficients.

BE_calc=np.array( ( 15.54*A - 16.94*A**(2/3) - 0.70*( Z**2 / A**(1/3) ) - 
         23.02*( (A-2*Z)**2 /A) + 12.50*delta/np.sqrt(A) ) * 10**3 / A ).reshape(-1,1) # keV / nucleon

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. Set up the regressor hyperparameters -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. 

SE_pysr_params=dict(
    # -- Search Space --
    maxsize=30, # for 5 terms.

    # -- Search Size --
    niterations=1000, # more interactions.
    populations=36,   # more islands.
    population_size=40,
    ncycles_per_iteration=350,

    # -- Objective and complexity --
    elementwise_loss="L2DistLoss()", # MSE standard.
    model_selection="best", # Pareto selection mode: Accuracy/Complexity.
    parsimony=0.001,
    complexity_of_constants=2, 
    adaptive_parsimony_scaling=5.0,
    # denoise=True,

    # -- Operator constraints --
    constraints={
        "+": (-1, -1),
        "*": (-1, -1),
        "-": (-1, -1),
        # "/": (-1, 1),   # denominator must be a leaf.
        "^": (-1, 1)   # exponent must be a leaf.   
    },
    nested_constraints={
        # "/":{"/":0},
        "^":{"^":0}
    },

    # -- Weight for mgenetic operators --
    # Favour rotation (cycle variables: a + ( b + c ) = () + c) and addition of nodes.
    # Reduce randomization (maintain the found structures)
    weight_add_node=3.0,
    weight_insert_node=0.5,          
    weight_delete_node=0.8,
    weight_do_nothing=0.05,
    weight_mutate_constant=0.05,
    weight_mutate_operator=0.3,
    weight_swap_operands=0.2,
    weight_rotate_tree=4.0,
    weight_randomize=0.05,          
    weight_simplify=0.002,
    weight_optimize=0.3,
    crossover_probability=0.15,

    # -- Simmulated Annealing --
    annealing=True,
    alpha=2.5,  # conservative exploration.
    perturbation_factor=0.1,
    probability_negate_constant=0.01,
    skip_mutation_failures=True,
    
    # -- Tourament selection --
    tournament_selection_n=12,
    tournament_selection_p=0.95,
    use_frequency=True,
    use_frequency_in_tournament=True,    

    # -- Optimización de constantes --
    # BFGS for (aV, aS, aC, aA, aP)
    optimizer_algorithm="BFGS",
    optimizer_nrestarts=5,
    optimize_probability=0.25,
    optimizer_iterations=15, # the more the better.
    should_optimize_constants=True,

    # -- Migration--
    fraction_replaced=0.0005,
    fraction_replaced_hof=0.07,
    migration=True,
    hof_migration=True,
    topn=12,    

    # -- Stopping criteria --
    max_evals=None,
    timeout_in_seconds=None,
    # early_stop_condition="f(loss, complexity) = (loss < 1000) ", # RMSE SEMF per nucleon = ~47 keV , RMSE PySR A = sqrt(1000) ~31 keV

    # -- Performance --
    parallelism="multithreading",    
    deterministic=False,
#    random_state=42, # with parallelism: "serial" and deterministic: True.
    batching=False,
    batch_size=64,
    precision=32,
    should_simplify=True,
    verbosity=1,
)

pysr_model=PySRRegressor(
    binary_operators=["+","-","*","^"], # "+","-","*","/","^"
    unary_operators=None,
    **SE_pysr_params,
    output_directory=r"C:\Users\Lauri\Desktop\TFG-code\results\models\trainA"
)

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. Fit the model -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-. Ready to start -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
pysr_model.fit(X,BE_calc)