import os

num_inducing = [50]
alpha = [2., 5., 10., 50., 1., 1e-1, 1e-2, 1e-3, 1e-6, 1e-12]

for current_num_inducing in num_inducing:
    for current_alpha in alpha:
        command = f'python -m docs.notebooks.exploration_conditionals_troubleshoot_schur_variance --num_inducing={current_num_inducing} --alpha={current_alpha}'
        os.system(command)