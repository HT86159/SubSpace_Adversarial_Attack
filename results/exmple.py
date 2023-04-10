from matplotlib_venn import venn2, venn2_circles
from matplotlib import pyplot as plt
figure, axes = plt.subplots(1, 2)
venn2(subsets={'10': 36, '01': 126, '11': 632}, set_labels = ('A', 'B'), ax=axes[0])
venn2_circles((32, 126, 632), ax=axes[0],linestyle='--', linewidth=0.8, color="black")

plt.savefig('vennss.png')
plt.show()
