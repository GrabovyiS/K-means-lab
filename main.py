import pandas as pd
from random import randint
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



def main():
    K = 3
    MAX_ITERATIONS = 100
    df = pd.read_csv('./iris.csv')

    # Scaling data
    scaler = MinMaxScaler()
    only_numbers = df.drop(['variety'], axis=1)
    normalized_data = scaler.fit_transform(only_numbers)

    new_df = pd.DataFrame(normalized_data, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
    new_df['variety'] = df.drop(['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], axis=1)
    new_df['cluster'] = 0
    
    df = new_df

    # Initial random centroids. Array stores indecies of dots in a df.
    rand1 = randint(0, 149)
    rand2 = randint(0, 149)
    rand3 = randint(0, 149)
    centr1 = {'sepal.length': df.loc[rand1]['sepal.length'], 
              'sepal.width': df.loc[rand1]['sepal.width'],
              'petal.length': df.loc[rand1]['petal.length'],
              'petal.width': df.loc[rand1]['petal.width']}
    
    centr2 = {'sepal.length': df.loc[rand2]['sepal.length'], 
              'sepal.width': df.loc[rand2]['sepal.width'],
              'petal.length': df.loc[rand2]['petal.length'],
              'petal.width': df.loc[rand2]['petal.width']}
    
    centr3 = {'sepal.length': df.loc[rand3]['sepal.length'], 
              'sepal.width': df.loc[rand3]['sepal.width'],
              'petal.length': df.loc[rand3]['petal.length'],
              'petal.width': df.loc[rand3]['petal.width']}
    
    centroids = [centr1, centr2, centr3]
    
    print(df)
    print(centroids, '\n\n')
    centroids = k_means(df, centroids, MAX_ITERATIONS)
    print(df)
    print(centroids)

    silhouette_avg = silhouette_score(df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']], df['cluster'])
    print("The average silhouette_score is :", silhouette_avg)

    plt.scatter(df['sepal.length'], df['sepal.width'], c=df['cluster'])
    plt.xlabel('sepal.length')
    plt.ylabel('sepal.width')
    plt.title('Clusterization of Iris dataset. Sepals.')
    plt.show()

    plt.scatter(df['petal.length'], df['petal.width'], c=df['cluster'])
    plt.xlabel('sepal.length')
    plt.ylabel('sepal.width')
    plt.title('Clusterization of Iris dataset. Petals.')
    plt.show()


def k_means(df, centroids, MAX_ITERATIONS):
    for i in range(MAX_ITERATIONS):
        clusterize(centroids, df)
        old_centroids = centroids
        centroids = get_new_centroids(df)

        if old_centroids == centroids:
            break

    return centroids
    

def get_new_centroids(df):
    cluster0 = {'sepal.length': 0, 
              'sepal.width': 0,
              'petal.length': 0,
              'petal.width': 0,
              'count': 0
              }
    
    cluster1 = {'sepal.length': 0, 
              'sepal.width': 0,
              'petal.length': 0,
              'petal.width': 0,
              'count': 0
              }
    
    cluster2 = {'sepal.length': 0, 
              'sepal.width': 0,
              'petal.length': 0,
              'petal.width': 0,
              'count': 0
              }
    
    # Get sums of values for each cluster
    for index, row in df.iterrows():
        if row['cluster'] == 0:
            cluster0['sepal.length'] += row['sepal.length']
            cluster0['sepal.width'] += row['sepal.width']
            cluster0['petal.length'] += row['petal.length']
            cluster0['petal.width'] += row['petal.width']
            cluster0['count'] += 1

        if row['cluster'] == 1:
            cluster1['sepal.length'] += row['sepal.length']
            cluster1['sepal.width'] += row['sepal.width']
            cluster1['petal.length'] += row['petal.length']
            cluster1['petal.width'] += row['petal.width']
            cluster1['count'] += 1

        if row['cluster'] == 2:
            cluster2['sepal.length'] += row['sepal.length']
            cluster2['sepal.width'] += row['sepal.width']
            cluster2['petal.length'] += row['petal.length']
            cluster2['petal.width'] += row['petal.width']
            cluster2['count'] += 1

    # Get averages that will become new centroids
    cluster0['sepal.length'] /= cluster0['count']
    cluster0['sepal.width'] /= cluster0['count']
    cluster0['petal.length'] /= cluster0['count']
    cluster0['petal.width'] /= cluster0['count']
        
    cluster1['sepal.length'] /= cluster1['count']
    cluster1['sepal.width'] /= cluster1['count']
    cluster1['petal.length'] /= cluster1['count']
    cluster1['petal.width'] /= cluster1['count']

    cluster2['sepal.length'] /= cluster2['count']
    cluster2['sepal.width'] /= cluster2['count']
    cluster2['petal.length'] /= cluster2['count']
    cluster2['petal.width'] /= cluster2['count']

    return [cluster0, cluster1, cluster2]



def clusterize(centroids, df):
    for index, row in df.iterrows():        
        # Assosiate the dot with the closer centroid
        cluster = get_closest_centroid_index(index, df, centroids)
        # Could count how many dots changed cluster here
        df.at[index, 'cluster'] = cluster


def get_closest_centroid_index(df_index, df, centroids):
    p1 = df.loc[df_index]['sepal.length']
    p2 = df.loc[df_index]['sepal.width']
    p3 = df.loc[df_index]['petal.length']
    p4 = df.loc[df_index]['petal.width']

    distances = []

    for i in range(3):
        q1 = centroids[i]['sepal.length']
        q2 = centroids[i]['sepal.width']
        q3 = centroids[i]['petal.length']
        q4 = centroids[i]['petal.width']
        distance = sqrt((p1 - q1)**2 + (p2 - q2)**2 + (p3 - q3)**2 + (p4 - q4)**2)

        distances.append({'index': i, 'distance': distance})
    
    closest = min(distances, key=lambda r: r['distance'])
    return(closest['index'])


main()