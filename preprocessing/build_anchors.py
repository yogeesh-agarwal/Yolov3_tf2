import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

def cluster_iou(current_box , clusters):
    x = np.minimum(clusters[: , 0] , current_box[0])
    y = np.minimum(clusters[: , 1] , current_box[1])

    intersection = x * y
    ious = intersection / ((current_box[0] * current_box[1]) + (clusters[: , 0] * clusters[:, 1]) - intersection)
    return ious

def kmeans(boxes, k, dist=np.median,seed=1):
    rows = boxes.shape[0]

    distances     = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed(seed)
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for icluster in range(k):
            distances[:,icluster] = 1 - cluster_iou(clusters[icluster], boxes)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters,nearest_clusters,distances

if __name__ == "__main__":
    train_data = None
    with open("../data/wider_training_data.pickle" , "rb") as f:
        train_data = pickle.load(f)

    boxes_wh = []
    kmax = 11
    for image_name in train_data:
        for object in train_data[image_name]:
            width = object[2]
            height = object[3]
            boxes_wh.append([width , height])
    boxes_wh = np.array(boxes_wh)
    dist = np.mean
    results = {}
    for k in range(2,kmax):
        clusters, nearest_clusters, distances = kmeans(boxes_wh,k,seed=2,dist=dist)
        WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
        result = {"clusters":             clusters,
                  "nearest_clusters":     nearest_clusters,
                  "distances":            distances,
                  "WithinClusterMeanDist": WithinClusterMeanDist}
        print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k,1-result["WithinClusterMeanDist"]))
        results[k] = result

    plt.figure(figsize=(6,6))
    plt.plot(np.arange(2,kmax),
             [1 - results[k]["WithinClusterMeanDist"] for k in range(2,kmax)],"o-")
    plt.title("within cluster mean of {}".format(dist))
    plt.ylabel("mean IOU")
    plt.xlabel("N clusters (= N anchor boxes)")
    plt.show()

    num_anchors = 9
    print(results[num_anchors]["clusters"])
