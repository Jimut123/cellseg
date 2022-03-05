"""
Plot the JSON files generated
"""

import matplotlib.pyplot as plt
import json


with open('history_da_150e.json', 'r') as f:
    history_da = json.load(f)
print (len(history_da))


# Plot the history here
source_image_loss_list = []
source_accuracy_list = []
source_domain_loss_list = []
target_domain_loss_list = []
dummy_count = 0
for sd_l, al, sdom_l, td_list in zip(history_da['source_image_loss'],  history_da['source_accuracy'], history_da['source_domain_loss'], history_da['target_domain_loss']):
    source_image_loss_list.append(history_da['source_image_loss'][str(dummy_count)])
    source_accuracy_list.append(history_da['source_accuracy'][str(dummy_count)])
    source_domain_loss_list.append(history_da['source_domain_loss'][str(dummy_count)])
    target_domain_loss_list.append(history_da['target_domain_loss'][str(dummy_count)])
    dummy_count += 1
    source_image_loss_list = []
source_accuracy_list = []
source_domain_loss_list = []
target_domain_loss_list = []
dummy_count = 0
for sd_l, al, sdom_l, td_list in zip(history_da['source_image_loss'],  history_da['source_accuracy'], history_da['source_domain_loss'], history_da['target_domain_loss']):
    source_image_loss_list.append(history_da['source_image_loss'][str(dummy_count)])
    source_accuracy_list.append(history_da['source_accuracy'][str(dummy_count)])
    source_domain_loss_list.append(history_da['source_domain_loss'][str(dummy_count)])
    target_domain_loss_list.append(history_da['target_domain_loss'][str(dummy_count)])
    dummy_count += 1


plt.figure(figsize=(12,6))
plt.title('Domain Adaptation Losses', fontsize=35, fontname = 'DejaVu Serif', fontweight = 500)
plt.plot(source_image_loss_list,color='brown', linestyle='--', dashes=(5, 1),  linewidth=3.0)
plt.plot(source_domain_loss_list,color='blue', linestyle='-', dashes=(5, 1),  linewidth=3.0)
plt.plot(target_domain_loss_list,color='red', linestyle='-.', dashes=(5, 1),  linewidth=3.0)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

lgd = plt.legend(['Source Classifier Loss', 'Feature Extractor Loss', 'Domain Classifier Loss'],loc="center right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.50, 0.67))
plt.savefig('da_plot_losses_history.eps',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('da_plot_losses_history.png',  bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(figsize=(12,6))
plt.title('Domain Adaptation Accuracy', fontsize=35, fontname = 'DejaVu Serif', fontweight = 500)
plt.plot(source_accuracy_list,color='green', linestyle='--', dashes=(5, 1),  linewidth=3.0)
#plt.plot(source_domain_loss_list,color='blue', linestyle='-', dashes=(5, 1),  linewidth=3.0)
#plt.plot(target_domain_loss_list,color='red', linestyle='-.', dashes=(5, 1),  linewidth=3.0)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

lgd = plt.legend(['Source Accuracy', 'Source Domain Loss', 'Target Domain Loss'],loc="lower right",
          prop={'family':'DejaVu Serif', 'size':20})#, bbox_to_anchor=(1.39, 0.86))
plt.savefig('da_plot_acc_history.eps',  bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig('da_plot_acc_history.png',  bbox_extra_artists=(lgd,), bbox_inches='tight')
