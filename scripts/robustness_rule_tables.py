import numpy as np

base_rule_tables = ["rule_tables/game_of_life.txt", "rule_tables/spore_life.txt"]
for base_rule_table in base_rule_tables:
    model = base_rule_table.split('/')[-1].split('.')[0]
    mode = "random"  # random/connected

    base_rules = np.loadtxt(base_rule_table, delimiter=',')
    states = base_rules.shape[0]
    changes = base_rules.shape[1]
    print(states, changes)

    if mode == "random":
        checked = []
        for ui in range(base_rules.flatten().shape[0]):
            base_copy = base_rules.flatten().copy()
            base_copy[ui] = (base_rules.flatten()[ui] + 1) % states
            rand_rules = base_copy.reshape((states, changes))
            # check to see if rule is a duplicate
            save_rule = True
            for obs_rule in checked:
                if np.array_equal(obs_rule, rand_rules):
                    save_rule = False
                    break
                
            if save_rule:
                np.savetxt(f"rule_tables/{model}_robustness/random_{ui}.txt",
                           rand_rules, delimiter=',', newline=",0\n", fmt='%d')
                checked.append(rand_rules)




        for di in range(base_rules.flatten().shape[0]):
            base_copy = base_rules.flatten().copy()
            base_copy[di] = (base_rules.flatten()[di] - 1) % states
            rand_rules = base_copy.reshape((states, changes))
            save_rule = True
            for obs_rule in checked:
                if np.array_equal(obs_rule, rand_rules):
                    save_rule = False
                    break
                
            if save_rule:
                np.savetxt(f"rule_tables/{model}_robustness/random_{ui}.txt",
                           rand_rules, delimiter=',', newline=",0\n", fmt='%d')
                checked.append(rand_rules)

    elif mode == "connected":
        rti = 0  # indexer for the filenames
        for si in range(states):
            val = -1  # tracks changes in values
            for ci in range(changes):
                if val == -1:
                    val = base_rules[si, ci]
                elif val != base_rules[si, ci]:
                    # change the previous value to the current one
                    base_copy = base_rules.copy()
                    base_copy[si, (ci-1) % changes] = base_copy[si, ci]
                    np.savetxt(f"rule_tables/{model}_robustness/connected_{rti}.txt",
                               base_copy, delimiter=',', fmt='%d')
                    rti += 1

                    # change the current value to the previous one
                    base_copy = base_rules.copy()
                    base_copy[si, ci] = val
                    np.savetxt(f"rule_tables/{model}_robustness/connected_{rti}.txt",
                               base_copy, delimiter=',', fmt='%d')
                    rti += 1
                    val = base_rules[si, ci]
