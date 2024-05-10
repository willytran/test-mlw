from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import plotly.express as px

def index(request):
  return render(request, 'mainapp/index.html')

def vis(request):
    customers_clusters = pd.read_csv("mainapp/data/CustomersClusters.csv")
    customers_clusters['Cluster'] = customers_clusters['Cluster'].astype(str)
    customers_clusters['AccountNumber'] = customers_clusters['AccountNumber'].astype(str)
    
    fig = px.scatter_3d(
        customers_clusters,
        x=customers_clusters['tsne_features1'],
        y=customers_clusters['tsne_features2'],
        z=customers_clusters['tsne_features3'],
        color=customers_clusters['Cluster'],
        color_discrete_sequence=px.colors.sequential.algae,
        hover_data=["AccountNumber"],
        opacity=1,
        title="Green-Got's Customers Clustering Visualization",
    )

    graph = fig.to_html(default_width="100%", default_height='750px')
    
    # Render template with figure
    return render(request, 'mainapp/vis.html', {'graph': graph})

def highlight(request):
    with open("mainapp/data/highlight.txt") as f:
      account_numbers = [line.strip() for line in f]

    customers_clusters = pd.read_csv("mainapp/data/CustomersClusters.csv")
    customers_clusters['Cluster'] = customers_clusters['Cluster'].astype(str)
    customers_clusters['AccountNumber'] = customers_clusters['AccountNumber'].astype(str)

    customers_clusters['IsAccount'] = customers_clusters['AccountNumber'].isin(account_numbers)
    
    fig = px.scatter_3d(
        customers_clusters,
        x=customers_clusters['tsne_features1'],
        y=customers_clusters['tsne_features2'],
        z=customers_clusters['tsne_features3'],
        color=customers_clusters.apply(lambda row: 'Highlighted accounts' if row['IsAccount'] else row['Cluster'], axis=1),
        color_discrete_sequence=px.colors.sequential.algae,
        hover_data=["AccountNumber"],
        opacity=1,
        title="Green-Got's Customers Clustering Visualization - Highlighted accounts",
    )

    graph = fig.to_html(default_width="100%", default_height='750px')
    
    return render(request, 'mainapp/highlight.html', {'graph': graph})

def stats(request):
    with open("mainapp/data/highlight.txt", "r") as file:
        accnumList = [line.strip() for line in file]
    lenList = len(accnumList)

    customers_clusters = pd.read_csv("mainapp/data/CustomersClusters.csv")
    customers_clusters['Cluster'] = customers_clusters['Cluster'].astype(str)
    customers_clusters['AccountNumber'] = customers_clusters['AccountNumber'].astype(str)
    customers_clusters['InTextList'] = customers_clusters['AccountNumber'].isin(accnumList)

    highlight_counts = customers_clusters.groupby('Cluster')['InTextList'].sum()
    context = {cluster: count for cluster, count in highlight_counts.items()}
    context["sum"] = sum(context.values())

    print(context)

    return render(request, 'mainapp/stats.html', {'context': context})