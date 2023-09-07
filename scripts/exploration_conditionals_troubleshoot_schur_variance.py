import os
num_inducing = [5, 10, 20, 50, 100]

for current_num_inducing in num_inducing:
    command = f'python -m docs.notebooks.exploration_conditionals_troubleshooting_schur_variance --num_inducing={current_num_inducing}'
    os.system(command)