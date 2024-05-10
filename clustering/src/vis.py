import plotly.express as px
import pandas as pd
from clustering import compute_kmeans 
from tsne import compute_tsne
from dataProcessing import process_data

def visualize_clusters(customers_clusters):

    customers_clusters['Cluster'] = customers_clusters['Cluster'].astype(str)
    customers_clusters['AccountNumber'] = customers_clusters['AccountNumber'].astype(str)

    fig = px.scatter_3d(
        customers_clusters,
        x=customers_clusters['tsne_features1'],
        y=customers_clusters['tsne_features2'],
        z=customers_clusters['tsne_features3'],
        color=customers_clusters['Cluster'],
        color_discrete_sequence=px.colors.sequential.Aggrnyl,
        hover_data=["AccountNumber"],
        opacity=1,
        title="Green-Got's Customers Clustering Visualization",
    )

    fig.update_layout(scene=dict(xaxis_title='Feature 1', yaxis_title='Feature 2', zaxis_title='Feature 3'))
    fig.show()
    fig.write_html("ClusteringVisualisationRFISAR.html")

if __name__ == '__main__':
    featuresData = process_data()
    if not featuresData.empty:
        tsne_data = compute_tsne(featuresData)
        customers_clusters = compute_kmeans(tsne_data)
        visualize_clusters(customers_clusters)
    else:
        print("No data found")
