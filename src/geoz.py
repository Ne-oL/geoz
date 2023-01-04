"""
Geographic Decision Zones (GeoZ)
==================================
GeoZ is a Python module integrating several machine learning algorithms to create Geographic Maps for the output of 
Unsupervised Machine Learning techniques. The module is geared mainly toward delineating the output from Clustering algorithms.
See (https://github.com/Ne-oL/geoz) for complete documentation (under construction).

"""

# Khalid ElHaj (2022)
# Geographic Decision Zones (GeoZ)
#
# A Library to convert Unsupervised Clustering Results into Geographical Maps
#
# Author: Khalid ElHaj <KG.Khair@Gmail.com>
#
# License: BSD 3 clause license

import math
import copy
import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
import geopandas as gpd
import shapely.geometry
import descartes


def convex_hull_plot(latlong, y_pred, grid_resolution=100, colormap='Set3'):

    '''
    This Function creates a Convex Hull for each set of points that belong to a distinct cluster using Shapely's "convex_hull" 
    (https://github.com/shapely/shapely) to eventually draw a map that contains all the clustered data. The usage of this method and 
    its main advantage is to detect any clear overlapping in the clustering algorithm as the other methods can draw overlapped regions.
    However, due to its geometrical nature, the method isn't capable of accurately delineating the clusters regions nor should it be 
    used for that. This method doesn't invlove any machine Learning Algorithms, thus it execute quickly and is suited for prototyping
    the clustering algorithm's parameter to a certain degree.
    
    Parameters
    ----------

    latlong : DataFrame
        The Latitude and Longitude Coordinates of the Data points. The DataFrame must contain two columns,
        These columns should be named 'LONGITUDE','LATITUDE' verbatim.
        
    y_pred : List
        The y_pred is the Clustering prediction of the samples submitted to the algorithm, the results need to be saved
        in a list array with a (-1) dimension.
        
    grid_resolution : int, default=100
        Number of grid points to use for plotting decision boundary. Higher values will make the plot look nicer but a bit 
        slower to render.
        
    colormap : str or Colormap, optional
        A Colormap instance or registered colormap name. The colormap maps the level values to colors. 
        Defaults to "Set3".  

    
    Returns
    -------   
    ax: matplotlib.axes._subplots.AxesSubplot
        The returned Object is an AxesSubplot, it will display automatically in IPython environments. the figure can also
        be called using the (.figure) attribute. this will allow the user to manipulate it or save it using matplotlib as backend.
                 
    '''


    # Create a list of Point objects for each data point
    data = pd.DataFrame({'X':latlong['LONGITUDE'], 'Y':latlong['LATITUDE'], 'cluster':y_pred})
    datagroup = data.groupby('cluster')
    count=datagroup.count()[['X']].rename(columns={'X':'Number of Points'})
    print('Note: Clusters containing less than 3 points will be removed as Convex hull needs 3 points Minimum to be drawn')
    print(count)
    data = datagroup.filter(lambda x: x.count().X >= 2)

    # Get the unique values in the cluster column
    unique_clusters = np.unique(data['cluster'])

    # Generate a list of RGBA tuples for the clusters
    colors = mpl.colormaps[colormap].colors
    color_map = dict(zip(unique_clusters, colors))
    fig, ax = plt.subplots()

    # Loop over the unique clusters
    for cluster in unique_clusters:
        # Select the data points belonging to the current cluster
        cluster_points = data[data['cluster'] == cluster][['X','Y']]
        cluster_points = np.array(cluster_points)

        # Create a MultiPoint object from the list of Point objects
        multipoint = shapely.geometry.MultiPoint(cluster_points)

        # Compute the convex hull of the data points
        convex_hull = multipoint.convex_hull
        
        # Get a reference to the current Axes object and add the convex hull as a PolygonPatch
        patch = descartes.PolygonPatch(convex_hull, fc=colors[cluster], alpha=0.5)
        ax.add_patch(patch)

    # Plot the data points on top of the convex hull
    plt.scatter(data['X'], data['Y'], c= data['cluster'], s= grid_resolution, cmap='viridis')
    fig.set_size_inches(grid_resolution, grid_resolution)

    return ax



def sklearn_plot(latlong, y_pred, C=100, gamma=30.0, grid_resolution=100, colormap='Set3', show_points=True, bazel=False, n_samples='default', extent=1, random_seed=None):
    '''
    This Function utilize Scikit-Learn's "DecisionBoundaryDisplay" (http://scikit-learn.org) to draw the map. 
    The advantage of this method is that it gives the user a lot of flexibility to modify and adjust the map 
    to his liking, as opposed to the other method in GeoZ library, this method is better suited for prototyping 
    and quick drafts as one of the option available to the user allows him to reduce the resolution, thus 
    produce more maps in a short amount of time.
    https://spatialreference.org/ref/epsg/
    
    Parameters
    ----------

    latlong : DataFrame
        The Latitude and Longitude Coordinates of the Data points. The DataFrame must contain two columns,
        These columns should be named 'LONGITUDE','LATITUDE' verbatim.
        
    y_pred : List
        The y_pred is the Clustering prediction of the samples submitted to the algorithm, the results need to be saved
        in a list array with a (-1) dimension.
        
    SVC Parameters:
    
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. 
            The penalty is a squared l2 penalty.
        
        gamma : {‘scale’, ‘auto’} or float, default=30
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                * if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
                * if ‘auto’, uses 1 / n_features
                * if float, must be non-negative.
            in the GeoZ library, the gamma parameter serves mainly as a controller to the buffer zone around the points of the 
            cluster, as well as the inerconnectedness of the distant points within the same cluster/class. Decreasing this 
            parameter would increase the buffer zone around the points and thus increase the uncertinity of the clusters boundaries. 
            decreasing it would decrease the buffer zone around the points and consequently decreaseing the uncertinity of the 
            clusters boundaries.
        
    grid_resolution : int, default=100
        Number of grid points to use for plotting decision boundary. Higher values will make the plot look nicer but be 
        slower to render.
        
    colormap : str or Colormap, optional
        A Colormap instance or registered colormap name. The colormap maps the level values to colors. 
        Defaults to "Set3".  
        
    show_points : bool, default True
        Display the points used to create the Decision Boundary. This would allow the user to check the accuracy of the model
        as well as any artifacts or missing clusters, thus alerting the user to enable bazel_cluster function.
        
    bazel : bool, default False
        This option allows the Function to creates a Bazel round the study area, the defaults are robust but not guaranteed to work.
        its pretty easy to check if the function worked or not by looking at the final map/Decision Boundary plot, if the map 
        have a Bazel, it means the method did NOT work and the user needs to adjust the default values to increase generated 
        data points to force the model to consider the Bazel cluster as the background. if it works, the map will be clear, 
        showing only the clusters of interest without the Bazel showing.
        
    n_samples : str or array (optional)
        This variable is generated based on the provided dataset, you only need to modify it if the method failed to force 
        SVM Classifier to consider the cluster as a background. The 'default' value takes the number of samples
        in the largest cluster and increase it by one and use it to create the bazel cluster.
        
    extent : int, default=1
        The "width" of the generated Bazel, there should be no need to change this variable, however, if the method
        failed after manipulating n_samples, The default can be increase to 2 or 3 while adjusting the n_samples
        accourdingly.    
        
    random_seed : int (optional)    
        Sets Numpy Random Seed as a constat for Reproducibility. This mainly affect the randomness of the Bazel samples
        distribution within the extent parameter.
    
    Returns
    -------   
    ax: matplotlib.axes._subplots.AxesSubplot
        The returned Object is an AxesSubplot, it will display automatically in IPython environments. the figure can also
        be called using the (.figure) attribute. this will allow the user to manipulate it or save it using matplotlib as backend.
                 
    '''
    
    
    fig, ax = plt.subplots()
    clf = SVC(C=C, gamma=gamma, random_state=random_seed)
    
    # Creating the Bazel
    X=latlong[['LONGITUDE','LATITUDE']]
    print('X: ',len(X))
    y=y_pred
    print('y: ',len(y))
    if bazel==True:
        X,y=bazel_cluster(X, y, n_samples=n_samples, extent=extent, random_seed=random_seed)
        print('X with Bazel: ',len(X))
        print('y with Bazel: ',len(y))
        X=X.to_numpy()
        y=np.array(y)
    else:
        X=X.to_numpy()
        y=np.array(y)
        
    clf.fit(X, y)
    print('Drawing Accuracy: ',clf.score(X, y)*100, '%\n\n')
    cmap = mpl.colormaps[colormap]
    
    DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', ax=ax ,grid_resolution=grid_resolution, plot_method='contourf',cmap=cmap)
    fig.set_size_inches(100,100)
    
    #Points for GroundTruth
    if show_points==True:
        #Convert the Points into Geometery file compatible with GeoPandas, then assign it correct coordinated and reproject it to similar projection to the shapefile (Basins)
        gdf = gpd.GeoDataFrame(latlong, geometry=gpd.points_from_xy(latlong.LONGITUDE, latlong.LATITUDE))
        gdf.plot( marker='o',  color='red', markersize=50, ax=ax)

    return ax

def mlx_plot(latlong, y_pred, C=100, gamma=30.0, bazel=False, n_samples='default', extent=1, random_seed=None):
    
    '''
    This Function utilize Mlxtend's "decision_regions" (http://rasbt.github.io/mlxtend/) to draw the map. The advantage of 
    this method is that it gives the user a very detailed map with high resolution, in addition to that, it uses different
    colors decision regions AND Sympoles for data points to represent different clusters. this shows great advantage over
    default color schemes used in Scikit-Learn (can be changed), as the number of clusters increase, the number of 
    colors used increase forcing the algorith to cycle through the same set and confusing the users, thus adding symploes
    to differentiate between clusters in addition to colors help significantly. however, the increased accuracy limits
    the usage of this method as it takes a lot of time to draw which is a considerable disadvantage when prototyping. 
    This plotting method should be the choice for the final exported map. 
    
    Parameters
    ----------

    latlong : DataFrame
        The Latitude and Longitude Coordinates of the Data points. The DataFrame must contain two columns,
        These columns should be named 'LONGITUDE','LATITUDE' verbatim.
        
    y_pred : List
        The y_pred is the Clustering prediction of the samples submitted to the algorithm, the results need to be saved
        in a list array with a (-1) dimension.
        
    SVC Parameters:
    
        C : float, default=1.0
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. 
            The penalty is a squared l2 penalty.
        
        gamma : {‘scale’, ‘auto’} or float, default=30
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
                * if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
                * if ‘auto’, uses 1 / n_features
                * if float, must be non-negative.
            in the GeoZ library, the gamma parameter serves mainly as a controller to the buffer zone around the points of the 
            cluster, as well as the inerconnectedness of the distant points within the same cluster/class. Decreasing this 
            parameter would increase the buffer zone around the points and thus increase the uncertinity of the clusters boundaries. 
            decreasing it would decrease the buffer zone around the points and consequently decreaseing the uncertinity of the 
            clusters boundaries.

    bazel : bool, default False
        This option allows the Function to creates a Bazel round the study area, the defaults are robust but not 
        guaranteed to work. its pretty easy to check if the function worked or not by looking at the 
        final map/Decision Boundary plot, if the map have a Bazel, it means the method did NOT work and the user needs
        to adjust the default values to increase generated data points to force the model to consider the Bazel cluster 
        as the background. if it works, the map will be clear, showing only the clusters of interest without the Bazel showing.
        
    n_samples : str or array (optional)
        This variable is generated based on the provided dataset, you only need to modify it if the method failed to force 
        SVM Classifier to consider the cluster as a background. The 'default' value takes the number of samples
        in the largest cluster and increase it by one and use it to create the bazel cluster.
        
    extent : int, default=1
        The "width" of the generated Bazel, there should be no need to change this variable, however, if the method failed 
        after manipulating n_samples, The default can be increase to 2 or 3 while adjusting the n_samples accourdingly.     
        
    random_seed : int (optional)    
        Sets Numpy Random Seed as a constat for Reproducibility. This mainly affect the randomness of the Bazel samples
        distribution within the extent parameter.
    
    Returns
    -------   
    ax: matplotlib.axes._subplots.AxesSubplot
        The returned Object is an AxesSubplot, it will display automatically in IPython environments. the figure can also
        be called using the (.figure) attribute. this will allow the user to manipulate it or save it using matplotlib as backend.
         
    '''

    fig, ax = plt.subplots()
    clf = SVC(C=C, gamma=gamma, random_state=random_seed)
    
    # Creating the Bazel
    X=latlong[['LONGITUDE','LATITUDE']]
    print('X: ',len(X))
    y=y_pred
    print('y: ',len(y))
    if bazel==True:
        X,y=bazel_cluster(X, y, n_samples=n_samples, extent=extent, random_seed=random_seed)
        print('X with Bazel: ',len(X))
        print('y with Bazel: ',len(y))
        X=X.to_numpy()
        y=np.array(y)
    else:
        X=X.to_numpy()
        y=np.array(y)
        
    clf.fit(X, y)
    print('Drawing Accuracy: ',clf.score(X, y)*100, '%\n\n')
    
    plot_decision_regions(X, y, clf=clf, legend=0, ax=ax)
    fig.set_size_inches(100,100)

    return ax
    
    
def bazel_cluster(X,y,n_samples='default', extent=1, random_seed=None):
    '''
    This Function creates a Bazel round the study area, the defaults are robust but not gurateed to work.
    its pretty easy to check if the function worked or not by looking at the final map/Decision Boundary plot,
    if the map have a Bazel, it means the method did NOT work and the user needs to adjust the default values to increase
    generated data points to force the model to consider the Bazel cluster as the background. if it works, the map will be
    clear, showing only the clusters of interest without the Bazel showing.
    
    Caution: This method have very specific usage to ONLY complement the SVM Classification algorithm by allowing all the 
    clusters to show on the Map through forcing the SVM Classifier to consider the generated dummy data as the majority, 
    thus acting as a background to the Decision Boundary feature space plot which is a representation of the geographic map.
    
    X: Dataset containing only Longitude and Latitude (must be the only features used by the Classifier to create the final map).
    
    y: array or array-like containing only the predicted clusters names (Must be Numerical e.g. 1,2,3,4...).
    
    n_samples: this variable is generated based on your dataset, you only need to modify it if the method fails force 
    SVM Classifier to consider the cluster as a background.
    
    extent: the "width" of the generated Bazel, there should be no need to change this variable, however, if the method
    failed after menipulating n_samples, you could try to increase the default to 2 or 3 while increase the n_samples
    accourdingly.
    
    Returns
    -------   
    X: pandas.DataFrame
        Dataset containing the Longitude and Latitude of the data points in addition to the extra points produced 
        by the algorithm that will act as the bazel for the SVM Classifier.
    y: list
        a list array containing the predicted clusters names (Numerical e.g. 1,2,3,4...) in addition to the extra cluster
        produced by the algorithm that will act as the bazel for the SVM Classifier.
    
    '''

    random.seed(a=random_seed)
   
    latExtra= X.LATITUDE.std()*extent
    longExtra= X.LONGITUDE.std()*extent
    
    LlatMin= X.LATITUDE.min()-X.LATITUDE.std()
    LlatMax= X.LATITUDE.max()+X.LATITUDE.std()
    LlongMin= X.LONGITUDE.min()-X.LONGITUDE.std()
    LlongMax= X.LONGITUDE.max()+X.LONGITUDE.std()
    
    UlatMin= X.LATITUDE.min()-latExtra
    UlatMax= X.LATITUDE.max()+latExtra
    UlongMin= X.LONGITUDE.min()-longExtra
    UlongMax= X.LONGITUDE.max()+longExtra
    
    bottom=[[LlatMin,UlatMin],(UlongMin,UlongMax)]
    top=[[LlatMax,UlatMax],(UlongMin,UlongMax)]
    left=[[UlatMin,UlatMax],(UlongMin,LlongMin)]
    right=[[LlatMin,UlatMax],(LlongMax,UlongMax)]
    sides=[bottom, top, left, right]
    lat=[]
    long=[]
    
    if n_samples=='default':
        n_samples=pd.Series(y).value_counts().values[0]
    
    for side in sides:
        for i in range(math.ceil(n_samples/4)):
            pointLat=random.uniform(side[0][0],side[0][1])
            pointLong=random.uniform(side[1][0],side[1][1])
            lat.append(pointLat)
            long.append(pointLong)
    dataset=pd.DataFrame({'LONGITUDE': long,'LATITUDE': lat})
    dataset['y']=np.array(y).max()+1
    #dataset['y']='bazel'
    Y=copy.deepcopy(y)
    Y.extend(dataset['y'].values)
    dataset=dataset[['LONGITUDE','LATITUDE']]
    X=pd.concat((X, dataset), axis=0,ignore_index= True)
    return X,Y
    
__all__ = [ "sklearn_plot", "mlx_plot", "convex_hull_plot"]