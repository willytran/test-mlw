import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

tsneData = pd.read_csv("../ComplianceMarketing/1804_tsneData.csv")
tsneData.drop(columns='AccountNumber')

model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(5,30), metric='calinski_harabasz', locate_elbow=True, timings=True
)

visualizer.fit(tsneData)
visualizer.show()