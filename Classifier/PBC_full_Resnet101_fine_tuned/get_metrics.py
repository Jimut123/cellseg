import numpy as np
import math
import glob

report_file = glob.glob('report_*.txt')

with open(report_file[0]) as f:
    lines = [line.rstrip() for line in f]

count = 0
precision_list = []
recall_list = []
f1_list = []
support_list = []
for line in lines:
    count += 1
    if count >= 3 and count <= 10:
        # print(count, line)
        
        all_values = str(line).strip().split(' ')
        #print(all_values)
        precision_list.append(float(all_values[7]))
        recall_list.append(float(all_values[13]))
        f1_list.append(float(all_values[19]))
        support_list.append(float(all_values[26]))

# print(precision_list)
# print(recall_list)
# print(f1_list)
# print(support_list)

weighted_prec_total = 0
weighted_recall_total = 0
weighted_f1_total = 0

for prec, rec, f1, sup in zip(precision_list,recall_list,f1_list,support_list):
    weighted_prec_total += prec*sup
    weighted_recall_total += rec*sup
    weighted_f1_total += f1*sup

print("Precision = ",weighted_prec_total/sum(support_list))
print("Recall = ",weighted_recall_total/sum(support_list))
print("F1 score = ",weighted_f1_total/sum(support_list))