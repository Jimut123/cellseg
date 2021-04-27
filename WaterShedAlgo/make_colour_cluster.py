from tqdm import tqdm
import pickle
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sub_folders = glob.glob('PBC_dataset_normal_DIB/*')

total_files_list = []
total_img_cnts = 0
for folders in sub_folders:
    print(folders)
    files_from_folders = glob.glob('{}/*'.format(folders))
    total_img_cnts += len(files_from_folders)
    print(len(files_from_folders))
    print("Cumulative = ", total_img_cnts)
    for files in files_from_folders:
        total_files_list.append(files)

print("Total img cnts. = ",total_img_cnts)

get_top_two = []
# A colour clustering based method
def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    get_top_two.append(colors)
    start = 0
    
    for (percent, color) in colors:
        #print(color, "{:0.2f}%".format(percent * 100))
        
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

# Load image and convert to a list of pixels
for item in tqdm(total_files_list):
    image = cv2.imread(total_files_list[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))
    # Find and display most dominant colors
    cluster = KMeans(n_clusters=5).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    # visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
    # plt.imshow(visualize)
    # plt.show()
    # cv2.imshow('visualize', visualize)
    # cv2.waitKey()
    # print(get_top_two)
file = open('colours_cluster.txt', 'wb')
pickle.dump(get_top_two, file)
file.close()




