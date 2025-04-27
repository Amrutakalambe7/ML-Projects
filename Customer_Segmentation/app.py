import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from kneed import KneeLocator

# Set Streamlit page config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Sidebar Section
st.sidebar.title("🛠 Explore Clustering Options")
st.sidebar.markdown("Welcome to **Customer Grouping Explorer**!")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    st.title("🎛️ Customer Segmentation App")
    st.write("**Preview of Uploaded Dataset**:")
    st.dataframe(dataset.head())

    feature_columns = dataset.columns.tolist()

    x_feature = st.sidebar.selectbox("Select X-axis feature", feature_columns, index=feature_columns.index('Annual Income (k$)') if 'Annual Income (k$)' in feature_columns else 0)
    y_feature = st.sidebar.selectbox("Select Y-axis feature", feature_columns, index=feature_columns.index('Spending Score (1-100)') if 'Spending Score (1-100)' in feature_columns else 1)

    X = dataset[[x_feature, y_feature]].values

    method = st.sidebar.selectbox("Select Clustering Method", ("K-Means", "Hierarchical Clustering"))

    # Elbow Method for KMeans to Suggest Best Cluster
    if method == "K-Means":
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Plot Elbow Graph
        st.subheader("Elbow Method - Optimal Clusters")
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss)
        ax.set_title("WCSS vs Number of Clusters")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        # Detecting the "elbow" in the plot
        k1 = KneeLocator(range(1, 11), wcss, curve="convex", direction="decreasing")
        suggested_clusters = k1.elbow if k1.elbow is not None else 5  # Default to 5 if no elbow is found

        st.sidebar.write(f"🔵 Suggested Optimal Clusters: **{suggested_clusters}**")

        n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, int(suggested_clusters))

        # Training KMeans model
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
        y_kmeans = kmeans.fit_predict(X)

        # Plotting Clusters
        st.subheader(f"K-Means Clustering with {n_clusters} Clusters")
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

        for cluster in range(n_clusters):
            ax.scatter(X[y_kmeans == cluster, 0], X[y_kmeans == cluster, 1], 
                       s=100, c=colors[cluster % len(colors)], label=f'Cluster {cluster+1}')
        
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   s=300, c='gold', marker='*', label='Centroids')
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title('Clusters of Customers (K-Means)')
        ax.legend()
        st.pyplot(fig)

    else:  # Hierarchical Clustering
        st.sidebar.write("📈 Building Dendrogram...")
        dendrogram_fig, dendrogram_ax = plt.subplots(figsize=(8, 5))
        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'), ax=dendrogram_ax)
        dendrogram_ax.set_title('Dendrogram')
        dendrogram_ax.set_xlabel('Customers')
        dendrogram_ax.set_ylabel('Euclidean distances')
        st.pyplot(dendrogram_fig)

        n_clusters = st.sidebar.slider("Select Number of Clusters", 2, 10, 5)

        # Training Hierarchical Clustering model
        hc = AgglomerativeClustering(n_clusters=n_clusters, metric ='euclidean', linkage='ward')
        y_hc = hc.fit_predict(X)

        # Plotting Clusters
        st.subheader(f"Hierarchical Clustering with {n_clusters} Clusters")
        fig, ax = plt.subplots()
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

        for cluster in range(n_clusters):
            ax.scatter(X[y_hc == cluster, 0], X[y_hc == cluster, 1], 
                       s=100, c=colors[cluster % len(colors)], label=f'Cluster {cluster+1}')
        
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title('Clusters of Customers (Hierarchical)')
        ax.legend()
        st.pyplot(fig)

else:
    st.title("📂 Please upload your customer dataset to begin!")
