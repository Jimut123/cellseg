
# https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv

import cv2, numpy as np
from sklearn.cluster import KMeans

"""
slide = 1

[117.6705       1.77719839 117.03801693] 3.84%
[172.58280915  94.090324   153.12470085] 4.39%
[194.82058077 177.46557351 167.10690332] 12.40%
[170.70230224 148.55876694 144.59796854] 17.21%
[214.20227361 207.26334012 188.8958876 ] 62.17%


slide = 2

[206.08948913  65.19376146 227.43069925] 3.53%
[223.78470398 115.46686206 230.58615337] 3.73%
[234.72720195 172.15449059 231.39208916] 4.92%
[232.47499642 223.68653559 219.5298394 ] 36.48%
[242.62847972 239.07792731 232.45217491] 51.35%


"""


def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        print(color, "{:0.2f}%".format(percent * 100))
        end = start + (percent * 300)
        cv2.rectangle(rect, (int(start), 0), (int(end), 50), \
                      color.astype("uint8").tolist(), -1)
        start = end
    return rect

# Load image and convert to a list of pixels
image = cv2.imread('slide_9.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reshape = image.reshape((image.shape[0] * image.shape[1], 3))

# Find and display most dominant colors
cluster = KMeans(n_clusters=5).fit(reshape)
visualize = visualize_colors(cluster, cluster.cluster_centers_)
visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)
cv2.imshow('visualize', visualize)
cv2.waitKey()
