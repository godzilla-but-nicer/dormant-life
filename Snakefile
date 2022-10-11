configfile: "workflow/config.yaml"


rule robustness_rule_tables:
    input:
        "rule_tables/game_of_life.txt",
        "rule_tables/spore_life.txt"
    output:
        expand("rule_tables/game_of_life_robustness/random_{i}.txt", i=range(config["robustness"]["gol_len"])),
        expand("rule_tables/spore_life_robustness/random_{i}.txt", i=range(config["robustness"]["sl_len"]))
    shell:
        "python scripts/robustness_rule_tables.py"          

rule generate_time_series:
    input:
        "scripts/generate_time_series.py"
    output:
        "data/transients/gol_transients.npy",
        "data/transients/sl_transients.npy",
        "data/time_series/gol_time_series.npz",
        "data/time_series/sl_time_series.npz",
        expand("data/transients/r{i}_2_transients.npy", i=range(config["sim_params"]["t_random"])),
        expand("data/random_rules/r{i}_2_rules.npy", i=range(config["sim_params"]["t_random"])),
        expand("data/transients/r{i}_3_transients.npy", i=range(config["sim_params"]["t_random"])),
        expand("data/random_rules/r{i}_3_rules.npy", i=range(config["sim_params"]["t_random"])),
        expand("data/time_series/sl_robust_{i}_time_series.npz", i=range(config["robustness"][sl_len])),
        expand("data/time_series/gol_robust_{i}_time_series.npz", i=range(config["robustness"][gol_len]))
    shell:
        "python scripts/generate_time_series.py "
        "{config[sim_params][runs]} "
        "{config[sim_params][max_steps]} "
        "{config[sim_params][t_grid]} "
        "{config[sim_params][t_random]} "
        "{config[sim_params][l_steps]} "
        "{config[sim_params][l_grid]} "
        "[config[robustness][sl_len]} "
        "[config[robustness][gol_len]} 


# should probably be rewriten to produce two seperate plots
rule living_cells_comparison:
    input:
        "data/time_series/gol_time_series.npz",
        "data/time_series/sl_time_series.npz",
        expand("data/time_series/sl_robust_{i}_time_series.npz", i=range(config["robustness"]["sl_len"])),
        expand("data/time_series/gol_robust_{i}_time_series.npz", i=range(config["robustness"]["gol_len"]))
    output:
        multiext("plots/living_compare/living_cells_compare", ".pdf", ".svg", ".png")
    shell:
        "python scripts/compare_alive_cells.py "
        "{config[living_cells][boot_num]} "
        "{config[living_cells][ci_perc]} "
        "{config[sim_params][runs]} "
        "{config[sim_params][l_steps]} "
        "{config[robustness][sl_len]} "
        "{config[robustness][gol_len]}"

rule living_cells_table_features:
    input:
        expand("data/time_series/sl_robust_{i}_time_series.npz", i=range(config["robustness"]["sl_len"])),
        expand("data/time_series/gol_robust_{i}_time_series.npz", i=range(config["robustness"]["gol_len"]))
    output:
        #multiext("plots/living_table_features/correlation_two_state", ".pdf", ".svg", ".png"),
        multiext("plots/living_table_features/correlation_three_state", ".pdf", ".svg", ".png")
    shell:
        "python scripts/living_table_features.py "
        "{config[sim_params][runs]} "
        "{config[sim_params][l_steps]} "
        "{config[robustness][sl_len]} "
        "{config[robustness][gol_len]}"