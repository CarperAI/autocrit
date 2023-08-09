import matplotlib
from matplotlib import pyplot
import numpy as np
import json

def generate_markdown_table(data):
    headers = data[0].keys()
    table = ' | '.join(headers) + '\n' + ' | '.join(['-' * len(header) for header in headers])

    for row in data:
        table += '\n' + ' | '.join([str(row[header]).strip().replace('\n', '<br>') for header in headers])

    return table

with open("artifacts/evaluation-StableBeluga-7B.json") as f:
    evaluation = json.load(f)

print(generate_markdown_table(evaluation[:8]))

avg_prior_scores = (1 - np.mean([x["prior_score"] for x in evaluation])) * 100
avg_after_scores = (1 - np.mean([x["after_score"] for x in evaluation])) * 100

textcolor = "#333"
matplotlib.style.use("ggplot")
matplotlib.rcParams.update({
    "font.family": "Futura",
    "font.size": 15,
    "text.color": textcolor,
    "axes.labelcolor": textcolor,
    "axes.labelpad": 12,
    "xtick.color": textcolor,
    "ytick.color": textcolor,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.titlesize": 14,
    "figure.figsize": (8, 8),
})

pyplot.bar([0, 0.5], [avg_prior_scores, avg_after_scores], width=0.25, color=["red", "#1f77b4"])
pyplot.yticks(np.arange(0, 110, 20))
pyplot.xticks([0, 0.5], ["prior", "finetuned"], fontsize=22)

ax = pyplot.gca()
ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=True, left=False, labelleft=True)
ax.set_facecolor("#fff")
ax.set_title(f"StableBeluga-7B on advbench/harmful_behaviors:eval", size=26, y=1.08, fontdict={"fontweight": "normal"})
ax.set_ylabel("% unsafe generations", size=26)
pyplot.tight_layout()
pyplot.savefig("artifacts/unsafe.png", dpi=120)

;
