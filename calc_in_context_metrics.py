from beans_zero.evaluate import compute_metrics
import json
import pandas as pd
import seaborn as sns

folders_list =  [
    "results\\in_context_dalls_first\\",
    "results\\in_context_spotted_first\\",
    "results\\in_context_random\\",
    "results\\just_list_random\\",
    "results\\just_list_random_3\\",
    "results\\just_list_random_all\\",
    "results\\just_list_random_trainver\\",
    "results\\just_list_random_trainver_3\\",
    "results\\just_list_random_trainver_all\\",
]

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
        
print(outputs_loaded["in_context_dalls_first"][0][0])

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
golden_ratio = 1.618
width = 20
height = width / golden_ratio  # ≈ 3.09
fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

i = 0
for df, filename in zip(metrics_per_setup.values(), metrics_per_setup.keys()):
    # if filename.endswith("after_tag.out"):
    #     continue
    xs = [x / 10 for x in range(0, 11)]
    accuracies = [metric['F1 Score'] for metric in df]
    
    ax.plot(xs, accuracies, label=filename, color=colors[i], **marker_kwargs)
    for xv, yv in zip(xs, accuracies):
        ax.annotate(
            f"{yv:.2f}", (xv, yv), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=16, color='black',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
        )
    i += 1

POP2 = 52
POP1 = 125 - POP2

# Add line for majority class
p1 = POP1 / (POP1 + POP2)
r1 = POP1 / POP1

p2 = 0
r2 = 0

f11 = 2 * (p1 * r1) / (p1 + r1)
f12 = 2 * (p2 * r2) / (p2 + r2 + 1e-10)
macro_f1 = (f11 + f12) / 2

plt.axhline(y=macro_f1, color='gray', linestyle='--', label='Majority class baseline', linewidth=1.5)

# Add line for random baseline
p1 = POP1 / (POP1 + POP2)
p2 = POP2 / (POP1 + POP2)

r1 = .5
r2 = .5

f11 = 2 * (p1 * r1) / (p1 + r1)
f12 = 2 * (p2 * r2) / (p2 + r2)
macro_f1 = (f11 + f12) / 2

ax.axhline(y=macro_f1, color='gray', linestyle='-.', label='Random baseline', linewidth=1.5)

ax.set_xlabel('Scaling factor', fontsize=24)
ax.set_ylabel('F1 Score', fontsize=24)
ax.grid(True, which='major', linestyle='-', linewidth=0.4, alpha=0.25)
ax.grid(True, which='minor', linestyle='-', linewidth=0.3, alpha=0.15)
ax.tick_params(axis='both', which='both', length=4, width=0.8, labelsize=24)
ax.legend(frameon=False, fontsize=24)
sns.despine(fig, ax,trim=True)
plt.savefig('f1_scaling_factors_in_context.png', bbox_inches='tight', dpi=300)