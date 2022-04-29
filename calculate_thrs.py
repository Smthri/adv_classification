import numpy as np

with open('class_successful_attack_amount.txt', 'r') as f:
    for line in f:
        distr = eval(line)
distr = np.array(distr)

thrs = np.full(len(distr), fill_value=0.1)
base_label = np.argmax(distr)

thrs *= (distr[base_label] / distr)
thrs = np.clip(thrs, 0.001, 0.5)
print(list(thrs))
