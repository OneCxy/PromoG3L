
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def calculate_gc_content(sequences):
    gc_contents = []
    for seq in sequences:
        gc_count = seq.count('G') + seq.count('C')
        total_count = len(seq)
        gc_content = gc_count / total_count if total_count > 0 else 0
        gc_contents.append(gc_content)
    return gc_contents


def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]


# Paths
natural_file = "datasets/threelevel/high.txt"
generated_file = "outputs/high20/result/RL_best/TXT/generated.txt"
random_file = "datasets/randoms/Ecoli_low20/Ecoli.txt"

generated_sequences = read_sequences_from_file(generated_file)
natural_sequences = read_sequences_from_file(natural_file)
random_sequences = read_sequences_from_file(random_file)

gc_generated = calculate_gc_content(generated_sequences)
gc_natural = calculate_gc_content(natural_sequences)
gc_random = calculate_gc_content(random_sequences)

data = {
    'Natural': gc_natural,
    'Generated': gc_generated,
    'Random': gc_random
}

# Plot
plt.figure(figsize=(10, 6))

sns.violinplot(
    data=list(data.values()),
    inner=None,
    bw_method='scott',
    density_norm='width',
    alpha=0.85
)

medians = [np.median(gc) for gc in data.values()]
q1 = [np.percentile(gc, 25) for gc in data.values()]
q3 = [np.percentile(gc, 75) for gc in data.values()]

ind = np.arange(len(data))

plt.vlines(ind, q1, q3, color='#444444', lw=2)
plt.scatter(
    ind,
    medians,
    color='white',
    edgecolor='#333333',
    marker='o',
    s=60,
    zorder=3
)

sns.boxplot(
    data=list(data.values()),
    color='black',
    boxprops=dict(edgecolor='black', linewidth=1),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    medianprops=dict(color='black', linewidth=1.5),
    width=0.08,
    fliersize=2
)

plt.xticks(ticks=ind, labels=data.keys(), fontsize=16)
plt.ylabel('GC Content', fontsize=16)
plt.tick_params(axis='both', labelsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(0.8)
    plt.gca().spines[axis].set_color('#444444')

# Export
output_path = 'outputs/high20/result/RL_best/high_gcviolin.png'

plt.savefig(output_path, dpi=300, bbox_inches='tight')

plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')

plt.savefig(
    output_path.replace('.png', '.tif'),
    dpi=300,
    bbox_inches='tight',
    format='tiff',
    pil_kwargs={"compression": "tiff_lzw"}
)

plt.close()

print(f"图表已保存至:")
print(f"PNG: {output_path}")
print(f"SVG: {output_path.replace('.png', '.svg')}")
print(f"TIF: {output_path.replace('.png', '.tif')}")
