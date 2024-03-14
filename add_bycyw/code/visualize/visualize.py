from matplotlib import pyplot as plt
import matplotlib
import os

save_path = "add_bycyw/code/visualize/pic"

cup =  {
    "Cabinet": 0.23085165859112933,
    "Fridge": 0.2219996272828923,
    "CounterTop": 0.2094670145359672,
    "DiningTable": 0.14899366380916884,
    "Sink": 0.08465337308982482,
    "Microwave": 0.0734718598583675,
    "Shelf": 0.01579388743943347,
    "SideTable": 0.01444278792396571,
    "Plate": 0.00018635855385762206,
    "Pan": 0.00013976891539321654
}
plt.figure(tight_layout=True,figsize=(1.7,1.4))
label = cup.keys()
data = cup.values()
plt.barh(range(len(label)),data)

# 把边框线去掉
ax = plt.gca()
# ax.xaxis.set_visible(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

# 把边框的空白去掉


y = range(len(label))
# plt.yticks(y, label)
plt.yticks(range(len(label)), label,fontsize=8)
plt.xticks([])  # 隐藏x轴的刻度线和标签
# plt.xlabel("Probability")
# plt.ylabel("Object")
plt.savefig(os.path.join(save_path, "cup_prob.png"),dpi=300,bbox_inches='tight')



