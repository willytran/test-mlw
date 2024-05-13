import pandas as pd
import numpy as np

with open("../rfisarAccounts3.txt", "r") as file:
    accnumList = [line.strip() for line in file]
    lenList = len(accnumList)

customers_clusters = pd.read_csv("../ComplianceMarketing/1804_customersClusters.csv")
customers_clusters['Cluster'] = customers_clusters['Cluster'].astype(str)
customers_clusters['AccountNumber'] = customers_clusters['AccountNumber'].astype(str)
customers_clusters['InTextList'] = customers_clusters['AccountNumber'].isin(accnumList)

highlight_counts = customers_clusters.groupby('Cluster')['InTextList'].sum()
for cluster, count in highlight_counts.items():
    print(f"Cluster {cluster}: {count} -", round(count*100/lenList, 2), "%")

customers_rfisar = customers_clusters[customers_clusters['AccountNumber'].isin(accnumList)]

# print(customers_rfisar)

mean_tsne_feature1 = customers_rfisar['tsne_features1'].mean()
mean_tsne_feature2 = customers_rfisar['tsne_features2'].mean()
mean_tsne_feature3 = customers_rfisar['tsne_features3'].mean()

print(mean_tsne_feature1, mean_tsne_feature2, mean_tsne_feature3)

mean_point = np.array([mean_tsne_feature1, mean_tsne_feature2, mean_tsne_feature3])

# calculate the distance between each point and the mean point
customers_clusters['distance'] = np.sqrt((customers_clusters['tsne_features1'] - mean_tsne_feature1) ** 2 + 
                                          (customers_clusters['tsne_features2'] - mean_tsne_feature2) ** 2 + 
                                          (customers_clusters['tsne_features3'] - mean_tsne_feature3) ** 2)

# sort the points based on their distance to the mean point
sorted_customers_clusters = customers_clusters.sort_values(by='distance')

# select the 10 closest points
closest_points = sorted_customers_clusters[~sorted_customers_clusters['AccountNumber'].isin(accnumList)].head(20)

print(closest_points[['AccountNumber', 'InTextList', 'distance', 'Cluster']])