from beans_zero.evaluate import compute_metrics
import json
import pandas as pd
import seaborn as sns
import numpy as np


def macro_f1_baselines(class_counts):
    """
    Given class counts, returns:
    (macro_f1_majority, macro_f1_uniform_random)
    """
    counts = np.array(class_counts, dtype=float)
    N = counts.sum()
    K = len(counts)

    # --- Majority-class model ---
    majority = np.argmax(counts)
    F1s_majority = []
    for i in range(K):
        if i == majority:
            TP = counts[i]
            FP = N - counts[i]
            FN = 0
        else:
            TP = 0
            FP = 0
            FN = counts[i]
        denom = 2*TP + FP + FN
        F1s_majority.append(0 if denom == 0 else 2*TP/denom)
    macro_f1_majority = np.mean(F1s_majority)

    # --- Uniform random model ---
    F1s_random = []
    for i in range(K):
        TP = counts[i]/K
        FP = (N - counts[i])/K
        FN = counts[i]*(1 - 1/K)
        denom = 2*TP + FP + FN
        F1s_random.append(0 if denom == 0 else 2*TP/denom)
    macro_f1_random = np.mean(F1s_random)

    return macro_f1_majority, macro_f1_random



folders_list =  [
    "results\\in_context_random\\",
    "results\\in_context_dalls_first\\",
    "results\\in_context_spotted_first\\",
    "results\\just_list_random\\",
    "results\\just_list_random_3\\",
    "results\\just_list_random_all\\",
    "results\\just_list_random_trainver\\",
    "results\\just_list_random_trainver_3\\",
    "results\\just_list_random_trainver_all\\",
    # "results\\just_classify\\",
    # "results\\just_classify_3\\",
    # "results\\just_classify_all\\",
    "results\\just_classify_explained\\",
    "results\\just_classify_explained_3\\",
    "results\\just_classify_explained_all\\",
    "results\\their_classification\\",
    "results\\their_classification_3\\",
    "results\\their_classification_all\\",
]

baselines = {}
outputs_loaded = dict()
for folder in folders_list:
    print(folder)
    name = folder.split("\\")[-2]
    outputs_loaded[name] = []
    for i in range(0, 11):
        scaling_output = []
        with open(folder + f"beans_zero_eval_unseen-family-cmn_query0_lora{i:02d}0.jsonl", "r") as f:
            for line in f:
                entry = json.loads(line)
                scaling_output.append((entry["prediction"], entry["label"]))
        outputs_loaded[name].append(scaling_output)
        if i == 0:
            # Compute baselines
            labels = [label for _, label in scaling_output]
            class_counts = pd.Series(labels).value_counts().sort_index().tolist()
            baselines[name] = macro_f1_baselines(class_counts)

metrics_per_setup = dict()

for dataset in outputs_loaded.keys():
    metrics_per_setup[dataset] = []
    for i in range(0, 11):
        outputs, labels = zip(*outputs_loaded[dataset][i])

        results = {"prediction": outputs, "label": labels, "dataset_name": ["unseen-species-cmn"] * len(outputs), "id": list(range(len(outputs)))}
        results_df = pd.DataFrame(results)
            
        metrics = compute_metrics(results_df, verbose=False)
        metrics_per_setup[dataset].append(metrics["unseen-species-cmn"])
        print(f"Dataset: {dataset}, Scaling: {i * 10}%")
        print(metrics["unseen-species-cmn"]["Accuracy"], metrics["unseen-species-cmn"]["F1 Score"])

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
# Publication-quality styling
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

marker_kwargs = dict(marker='o', markersize=4, markerfacecolor='white', markeredgewidth=1.5)
colors = sns.color_palette("Set2", n_colors=len(folders_list))
# Golden ratio proportions for aesthetically pleasing layout

tests = list(metrics_per_setup.keys())
threes = [name for name in tests if name.endswith("_3")]
alls = [name for name in tests if name.endswith("_all")]
tests = [name for name in tests if not (name.endswith("_3") or name.endswith("_all"))]
in_contexts = [name for name in tests if "in_context" in name]
tests = [name for name in tests if "in_context" not in name or "random" in name]

name_maps = {
    "in_context_dalls_first": "In-context (Dall's first)",
    "in_context_spotted_first": "In-context (Spotted first)",
    "in_context_random": "In-context (Random order)",
    "just_list_random_trainver": "Detection + Classes†",
    "just_list_random": "Classification + Classes",
    "just_classify": "Classification",
    "just_classify_explained": "Classification",
    "their_classification": "Classification†",
}

for names, kind, title in [(tests, "", "2 Classes"), (threes, "_3", "3 Classes"), (alls, "_all", "All Classes"), (in_contexts, "_order_check", "2 Classes with Different Orders")]:
    golden_ratio = 1.618
    width = 20
    height = width / golden_ratio  # ≈ 3.09
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    i = 0
    for df, filename in zip(metrics_per_setup.values(), metrics_per_setup.keys()):
        if not filename in names:
            continue
        # if filename.endswith("after_tag.out"):
        #     continue
        xs = [x / 10 for x in range(0, 11)]
        accuracies = [metric['F1 Score'] for metric in df]
        
        line_style = '-'

        for key, value in name_maps.items():
            if key in filename:
                filename = value
                break
        
        ax.plot(xs, accuracies, label=filename, color=colors[i], linestyle=line_style, **marker_kwargs)
        for xv, yv in zip(xs, accuracies):
            ax.annotate(
                f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=16, color='black',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            )
        i += 1


    # Add line for majority class
    macros = baselines[names[0]]
    plt.axhline(y=macros[0], color="grey", linestyle='--', label=f'Majority class baseline', linewidth=1.5)

    ax.axhline(y=macros[1], color="grey", linestyle='-.', label=f'Random baseline', linewidth=1.5)

    ax.set_ylim(0, 1)
    ax.set_xlabel('Scaling factor', fontsize=24)
    ax.set_ylabel('F1 Score', fontsize=24)
    ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
    ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
    ax.legend(frameon=False, fontsize=24)
    sns.despine(fig, ax,trim=True)
    plt.title(title, fontsize=28)
    plt.savefig(f'f1_scaling_factors_in_context{kind}.png', bbox_inches='tight', dpi=300)
    plt.clf()