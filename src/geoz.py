"""
Geographic Decision Zones (GeoZ)
==================================
GeoZ is a Python module integrating several machine learning algorithms to create Geographic Maps for the output of 
Unsupervised Machine Learning techniques. The module is geared mainly toward delineating the output from Clustering algorithms.
See (https://github.com/Ne-oL/geoz) for complete documentation (under construction).

"""

# Khalid ElHaj (2024)
# Geographic Decision Zones (GeoZ)
#
# A Library to convert Unsupervised Clustering Results into Geographical Maps
#
# Author: Khalid ElHaj <KG.Khair@Gmail.com>
#
# License: BSD 3 clause license

# Standard library imports
import math
import copy
import random

# Third-party library imports
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.svm import SVC
from scipy.spatial import Voronoi
from sklearn.inspection import DecisionBoundaryDisplay
from mlxtend.plotting import plot_decision_regions

# Matplotlib-specific imports
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path

# Shapely-specific imports
from shapely.geometry import Polygon as ShapelyPolygon, MultiPoint
from shapely.ops import unary_union

# Global options
GEOZ_COLOR_SET = 'tab20'
GEOZ_FIGURE_RESOLUTION = 100

def set_geoz_options(color_set=None, figure_resolution=None):
    """
    Set global options for GeoZ visualizations.

    This function allows users to modify global settings for all GeoZ plotting functions.
    These settings include the color scheme and the base figure resolution.

    Parameters:
    -----------
    color_set : str, optional
        The name of a matplotlib colormap to use for visualizations. This should be a 
        discrete colormap (e.g., 'tab20', 'Set1', 'Paired') as the functions are designed 
        for categorical data, not continuous. If not specified or invalid, the default 
        'tab20' colormap is used.

    figure_resolution : int, optional
        The base resolution to use for figure sizes. The actual figure size is determined 
        by multiplying this value by the aspect ratio of the data's spatial distribution. 
        If not specified, the default value of 100 is used.

    Notes:
    ------
    - The aspect ratio of the figures is automatically calculated based on the 
      spatial distribution of the input data points. This ensures that the 
      geographical shapes are not distorted.

    - The color_set parameter only accepts discrete colormaps. Continuous colormaps 
      may fail not render correctly in the plotting functions (especially voronoi_regions_plot function).

    - These global options affect all subsequent calls to GeoZ plotting functions 
      (sklearn_plot, convex_hull_plot, voronoi_regions_plot) except (mlx_plot) as it uses its own colorscheme and symbols.
      These global options are effictive unless overridden in the individual function calls.

    - The default values are:
        color_set: 'tab20'
        figure_resolution: 100
        
    - GEOZ_FIGURE_RESOLUTION vs. grid_resolution:
      * GEOZ_FIGURE_RESOLUTION (set here) determines the overall size and quality of 
        the output figure. It affects the physical dimensions of the plot.
      * grid_resolution (found in some functions like sklearn_plot) determines the 
        density of the grid used for decision boundary calculations. It affects the 
        smoothness and detail of the decision boundaries, not the figure size.
      These two parameters serve different purposes and can be adjusted independently.

    Examples:
    ---------
    >>> import geoz
    >>> geoz.set_geoz_options(color_set='Set1', figure_resolution=200)
    >>> # Now all subsequent plotting function calls will use these settings
    >>> geoz.sklearn_plot(latlong, y_pred, grid_resolution=150)
    # In this example, the figure size is based on figure_resolution=200,
    # while the decision boundary detail is based on grid_resolution=150

    Warnings:
    ---------
    - If an invalid color_set is provided, a warning is printed and the default is used.
    - No warning is given for invalid figure_resolution; it's the user's responsibility 
      to provide a reasonable positive integer.
    """
    
    global GEOZ_COLOR_SET, GEOZ_FIGURE_RESOLUTION
    if color_set is not None:
        if color_set in mpl.colormaps and isinstance(mpl.colormaps[color_set], mpl.colors.ListedColormap):
            GEOZ_COLOR_SET = color_set
        else:
            print(f"Warning: '{color_set}' is not a valid discrete matplotlib colormap. Using default 'tab20'.")
    if figure_resolution is not None:
        if isinstance(figure_resolution, int) and figure_resolution > 0:
            GEOZ_FIGURE_RESOLUTION = figure_resolution
        else:
            print(f"Warning: figure_resolution must be a positive integer. Using default value of 100.")


def convex_hull_plot(latlong, y_pred, grid_resolution=100, show_points=True, colormap=None):

    '''
    This Function creates a Convex Hull for each set of points that belong to a distinct cluster using Shapely's "convex_hull" 
    (https://github.com/shapely/shapely) to eventually draw a map that contains all the clustered data. 
    The usage of this method and its main advantage is to detect any clear overlapping in the clustering algorithm as the other methods can draw overlapped regions.
    However, due to its geometrical nature, the method isn't capable of accurately delineating the clusters regions nor should it be used for that. 
    This method doesn't invlove any machine Learning Algorithms, thus it execute quickly and is suited for prototyping
    the clustering algorithm's parameter to a certain degree.
    
    Parameters
    ----------

    latlong : DataFrame
        The Latitude and Longitude Coordinates of the Data points. The DataFrame must contain two columns in this exact order.
        The First column should contain the Latitude data while the second column should contain the the Longitude data.
        
    y_pred : List
        The y_pred is the Clustering prediction of the samples submitted to the algorithm, the results need to be saved
        in a list array with a (-1) dimension.
        
    grid_resolution : int, default=100
        Number of grid points to use for plotting decision boundary. Higher values will make the plot look nicer but a bit 
        slower to render.

    show_points : bool, default True
        Display the points used to create the Convex Hull diagram. This would allow the user to check the accuracy of the model
        as well as any artifacts or missing clusters.
        
    colormap : str or Colormap, optional
        A Colormap instance or registered colormap name. The colormap maps the level values to colors. 
        Defaults to "tab20".  

    
    Returns
    -------   
    ax: matplotlib.axes._subplots.AxesSubplot
        The returned Object is an AxesSubplot, it will display automatically in IPython environments. the figure can also
        be called using the (.figure) attribute. this will allow the user to manipulate it or save it using matplotlib as backend.
                 
    Notes
    -----
    - Clusters containing fewer than 3 points will be removed as a Convex hull requires a minimum of 3 points to be drawn.
    - This function uses Shapely for geometry operations and matplotlib for plotting.
    
    '''

    if colormap is None:
        colormap = GEOZ_COLOR_SET
    
    lon_col = latlong.columns[1]
    lat_col = latlong.columns[0]
    X=latlong[[lon_col, lat_col]]

    # Create a list of Point objects for each data point
    data = pd.DataFrame({'X':latlong[lon_col], 'Y':latlong[lat_col], 'cluster':y_pred})
    datagroup = data.groupby('cluster')
    count=datagroup.count()[['X']].rename(columns={'X':'Number of Points'})
    data = datagroup.filter(lambda x: x.count().X >= 2)

    # Get the unique values in the cluster column
    unique_clusters = np.unique(data['cluster'])

    # Generate a list of RGBA tuples for the clusters
    colors = mpl.colormaps[colormap].colors
    unique_clusters = np.unique(y_pred)
    color_map = dict(zip(range(len(unique_clusters)), colors[:len(unique_clusters)]))
    cluster_to_index = {cluster: index for index, cluster in enumerate(unique_clusters)}
    
    # Calculate the Figure Dimensions        
    latRange = latlong[lat_col].max() - latlong[lat_col].min()
    lonRange = latlong[lon_col].max() - latlong[lon_col].min()
    ratio = lonRange/latRange

    fig, ax = plt.subplots(figsize=(GEOZ_FIGURE_RESOLUTION * ratio, GEOZ_FIGURE_RESOLUTION))

    # Loop over the unique clusters
    for cluster in unique_clusters:
        # Select the data points belonging to the current cluster
        cluster_points = data[data['cluster'] == cluster][['X','Y']]
        cluster_points = np.array(cluster_points)

        # Create a MultiPoint object from the list of Point objects
        multipoint = MultiPoint(cluster_points)

        # Compute the convex hull of the data points
        convex_hull = multipoint.convex_hull
        
        # Get a reference to the current Axes object and add the convex hull as a PolygonPatch
        if convex_hull.geom_type == 'Polygon':
            coords = list(convex_hull.exterior.coords)
            patch = Polygon(coords, fc=color_map[cluster_to_index[cluster]], alpha=0.5)
            ax.add_patch(patch)

    # Plot the data points on top of the convex hull
    if show_points == True:
        plt.scatter(data['X'], data['Y'], c=[cluster_to_index[c] for c in data['cluster']], s=grid_resolution, cmap=colormap)
    else:
        plt.scatter(data['X'], data['Y'], c='None', s=grid_resolution, cmap=colormap)
    
    fig.set_size_inches(GEOZ_FIGURE_RESOLUTION * ratio, GEOZ_FIGURE_RESOLUTION)

    return ax



def sklearn_plot(latlong, y_pred, C=100, gamma=30.0, grid_resolution=100, colormap=None, crs=None, show_points=False, bazel=False, n_samples='default', extent=1, ax=None, alpha=None, random_seed=None):
    
    '''
    This Function utilize Scikit-Learn's "DecisionBoundaryDisplay" (http://scikit-learn.org) to draw the map. 
    The advantage of this method is that it gives the user a lot of flexibility to modify and adjust the map 
    to his liking, as opposed to the other method in GeoZ library, this method is better suited for prototyping 
    and quick drafts as one of the option available to the user allows him to reduce the resolution, thus 
    produce more maps in a short amount of time.
    
    Parameters
    ----------

    latlong : DataFrame
        The Latitude and Longitude Coordinates of the Data points. The DataFrame must contain two columns in this exact order.
        The First column should contain the Latitude data while the second column should contain the the Longitude data.
        
    y_pred : List (Pandas.Series)
        The y_pred is the Clustering prediction of the samples submitted to the algorithm, the results need to be saved
        in a list array with a (-1) dimension.
        
    SVC Parameters:
    
        C : float, default=100
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
        Defaults to "tab20".
        
    crs : str, optional
        Coordinate Reference System of the Study Area you want to model. if you do not know yours, you can visit the 
        EPSG website (https://spatialreference.org/ref/epsg/) and search for the appropriate authority string for your area, 
        it should be a string like (eg "EPSG:4326") or a WKT string.

    show_points : bool, default True
        Display the points used to create the Decision Boundary. This would allow the user to check the accuracy of the model
        as well as any artifacts or missing clusters, thus alerting the user to enable bazel_cluster function.
        
    bazel : bool, default False
        This option allows the Function to creates a Bazel round the study area, the defaults are robust but not guaranteed to work.
        its pretty easy to check if the function worked or not by looking at the final map/Decision Boundary plot, if the map 
        have a Bazel, it means the method did NOT work and the user needs to adjust the default values to increase generated 
        data points to force the model to consider the Bazel cluster as the background. if it works, the map will be clear, 
        showing only the clusters of interest without the Bazel showing.
        
    n_samples : int, optional
        This variable is generated based on the provided dataset, you only need to modify it if the method failed to force 
        SVM Classifier to consider the synthetic cluster as a background. The 'default' value takes the number of samples
        in the largest cluster and increase it by one and use it to create the synthetic bazel cluster.
        
    extent : int, default=1
        The "width" of the generated Bazel, there should be no need to change this variable, however, if the method
        failed after manipulating n_samples, The default can be increase to 2 or 3 while adjusting the n_samples
        accourdingly.    

    ax : matplotlib.axes.Axes, optional
        An existing matplotlib Axes. If provided, the function will draw on this existing axis
        instead of creating a new figure. In this case, the function may not automatically
        display the output. Users may need to manually display the existing figure or
        access the figure via the returned ax object's .figure attribute.
        
    alpha : float, optional
        sets the transparency of the resulting figure. This is enabled only if the ax paremeter is utilized 
        as it was surfaced to allow the underlying ax figure to show along the created decision boundaries.
        
    random_seed : int, optional  
        Sets Numpy Random Seed as a constat for Reproducibility. This mainly affect the randomness of the Bazel samples
        distribution within the extent parameter.
    
    Returns
    -------   
    ax: matplotlib.axes._subplots.AxesSubplot
        The returned Object is an AxesSubplot, it will display automatically in IPython environments. the figure can also
        be called using the (.figure) attribute. this will allow the user to manipulate it or save it using matplotlib as backend.
                 
    Notes
    -----
    This function uses Scikit-learn's SVC (Support Vector Classification) for creating
    decision boundaries and matplotlib for visualization. The SVC is fitted on the
    provided data points and then used to predict over a grid, creating the decision
    boundary plot.

    The 'bazel' option creates a synthetic background cluster, which can be useful
    for forcing the SVC to consider the area outside your clusters as a separate class.
    If enabled, check the resulting plot to ensure the bazel isn't visible, which would
    indicate successful background separation.

    The colormap choice can significantly impact the readability of your plot, especially
    with many clusters. Consider using a colormap that provides good distinction between
    colors for your number of clusters.

    When using an existing axis (ax parameter), the transparency option (alpha) allows
    for overlaying the decision boundaries on existing plots, which can be useful for
    comparing different clustering results or overlaying on geographical features.

    This method typically produces smoother decision boundaries compared to the mlx_plot
    function, but may be less precise for complex cluster shapes. It's generally faster
    for smaller datasets but may become slower for very large datasets.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data
    >>> np.random.seed(0)
    >>> latlong = pd.DataFrame({
    ...     'LONGITUDE': np.random.uniform(-180, 180, 100),
    ...     'LATITUDE': np.random.uniform(-90, 90, 100)
    ... })
    >>> y_pred = np.random.randint(0, 5, 100)
    >>> # Create the plot
    >>> ax = sklearn_plot(latlong, y_pred)
    >>> ax.set_title("Sample Cluster Plot")
    '''
    
    # Creating the Bazel
    lon_col = latlong.columns[1]
    lat_col = latlong.columns[0]
    X=latlong[[lon_col, lat_col]]
    
    # Calculate the Figure Dimensions        
    latRange = latlong[lat_col].max() - latlong[lat_col].min()
    lonRange = latlong[lon_col].max() - latlong[lon_col].min()
    ratio = lonRange/latRange    
    
    print('Total Number of Samples:')
    print('X: ',len(X))
    y=y_pred
    print('y: ',len(y))
    if bazel==True:
        print('\nBazel Function is: \033[91m\033[1mEnabled\033[0m \nIf the Bazel Cluster is visible, this indicates that the function has failed, try to modify the number of samples or the extent of the Bazel and run it again \n')
        X,y=bazel_cluster(X, y, n_samples=n_samples, extent=extent, random_seed=random_seed)
        print('X with Bazel: ',len(X))
        print('y with Bazel: ',len(y))
        X=X.to_numpy()
        y=np.array(y)
    else:
        X=X.to_numpy()
        y=np.array(y)
    
    if colormap is None:
        colormap = GEOZ_COLOR_SET
        
    if ax==None:
        fig, ax = plt.subplots()
        alpha=1
    else:
        if alpha==None:
            alpha=0.7
    
    clf = SVC(C=C, gamma=gamma, random_state=random_seed)
    clf.fit(X, y.ravel())
    print('\nDrawing Accuracy: ',np.round(clf.score(X, y)*100,2), '%\n\n')
    cmap = mpl.colormaps[colormap]
    
    DecisionBoundaryDisplay.from_estimator(clf, X, response_method='predict', ax=ax ,grid_resolution=grid_resolution, alpha=alpha, plot_method='contourf',cmap=cmap)
    ax.figure.set_size_inches(GEOZ_FIGURE_RESOLUTION * ratio, GEOZ_FIGURE_RESOLUTION)
    
    #Points for GroundTruth
    if show_points==True:
        #Convert the Points into Geometery file compatible with GeoPandas, then assign it correct coordinated and reproject it to similar projection to the shapefile (Basins)
        gdf = gpd.GeoDataFrame(latlong, geometry=gpd.points_from_xy(latlong[lon_col], latlong[lat_col], crs=crs))
        gdf.plot( marker='o',  color='red', markersize=50, ax=ax)

    return ax



def mlx_plot(latlong, y_pred, C=100, gamma=30.0, bazel=False, n_samples='default', extent=1, ax=None, random_seed=None, n_jobs=None):
    
    '''
    This Function utilize Mlxtend's "decision_regions" (http://rasbt.github.io/mlxtend/) to draw the map. The advantage of 
    this method is that it gives the user a very detailed map with high resolution, in addition to that, it uses different
    colors decision regions AND Symbols for data points to represent different clusters. this shows great advantage over
    default color schemes used in Scikit-Learn (can be changed), as the number of clusters increase, the number of 
    colors used increase forcing the algorithm to cycle through the same set and confusing the users, thus adding symploes
    to differentiate between clusters in addition to colors help significantly. however, the increased accuracy limits
    the usage of this method as it takes a lot of time to draw which is a considerable disadvantage when prototyping. 
    This plotting method should be the choice for the final exported map. 
    
    Parameters
    ----------

    latlong : DataFrame
        The Latitude and Longitude Coordinates of the Data points. The DataFrame must contain two columns in this exact order.
        The First column should contain the Latitude data while the second column should contain the the Longitude data.
        
    y_pred : List (Pandas.Series)
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
        
    n_samples : int, optional
        This variable is generated based on the provided dataset, you only need to modify it if the method failed to force 
        SVM Classifier to consider the cluster as a background. The 'default' value takes the number of samples
        in the largest cluster and increase it by one and use it to create the bazel cluster.
        
    extent : int, default=1
        The "width" of the generated Bazel, there should be no need to change this variable, however, if the method failed 
        after manipulating n_samples, The default can be increase to 2 or 3 while adjusting the n_samples accourdingly.     
        
    ax : matplotlib.axes.Axes, optional
        An existing matplotlib Axes. If provided, the function will draw on this existing axis
        instead of creating a new figure. In this case, the function may not automatically
        display the output. Users may need to manually display the existing figure or
        access the figure via the returned ax object's .figure attribute.
        
    random_seed : int, optional   
        Sets Numpy Random Seed as a constat for Reproducibility. This mainly affect the randomness of the Bazel samples
        distribution within the extent parameter.
        
    n_jobs : int, default=None
        The number of CPU Cores utilized to do the computation using Python's multiprocessing library.
        `None` means 1, and `-1` means using all processors.
    
    Returns
    -------   
    ax: matplotlib.axes._subplots.AxesSubplot
        The returned Object is an AxesSubplot, it will display automatically in IPython environments. the figure can also
        be called using the (.figure) attribute. this will allow the user to manipulate it or save it using matplotlib as backend.

    Notes
    -----
    This function uses Mlxtend's decision_regions for plotting, providing color-coded
    regions and distinct symbols for data points. This dual representation is particularly
    useful for distinguishing between a large number of clusters.

    The resulting plot typically has higher resolution compared to sklearn-based methods,
    offering more detailed and accurate boundary representations. However, this increased
    detail results in longer computation times, especially for large datasets or high
    numbers of clusters.

    The 'bazel' option creates a synthetic background cluster, useful for geographical
    data to clearly delineate the area of interest from surrounding regions. If enabled,
    check that the bazel is not visible in the final plot, indicating successful background separation.

    The `n_jobs` parameter enables parallel processing, potentially speeding up computation
    for large datasets. Use caution with `n_jobs=-1` on machines with limited resources to
    avoid system slowdowns.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data
    >>> np.random.seed(0)
    >>> latlong = pd.DataFrame({
    ...     'LONGITUDE': np.random.uniform(-180, 180, 100),
    ...     'LATITUDE': np.random.uniform(-90, 90, 100)
    ... })
    >>> y_pred = np.random.randint(0, 5, 100)
    >>> # Create the plot
    >>> ax = mlx_plot(latlong, y_pred)
    >>> ax.set_title("Sample Cluster Plot")
    '''
    
    clf = SVC(C=C, gamma=gamma, random_state=random_seed)
    
    # Creating the Bazel
    lon_col = latlong.columns[1]
    lat_col = latlong.columns[0]
    X=latlong[[lon_col,lat_col]]
    print('Total Number of Samples:')
    print('X: ',len(X))
    y=y_pred
    print('y: ',len(y))
    if bazel==True:
        print('\nBazel Function is: \033[91m\033[1mEnabled\033[0m \nIf the Bazel Cluster is visible, this indicates that the function has failed, try to modify the number of samples or the extent of the Bazel and run it again \n')        
        X,y=bazel_cluster(X, y, n_samples=n_samples, extent=extent, random_seed=random_seed)
        print('X with Bazel: ',len(X))
        print('y with Bazel: ',len(y))
        X=X.to_numpy()
        y=np.array(y)
    else:
        X=X.to_numpy()
        y=np.array(y)

    # Calculate the Figure Dimensions        
    latRange = latlong[lat_col].max() - latlong[lat_col].min()
    lonRange = latlong[lon_col].max() - latlong[lon_col].min()
    ratio = lonRange/latRange

    if ax==None:
        fig, ax = plt.subplots()

    clf.fit(X, y.ravel())
    print('\nDrawing Accuracy: ',np.round(clf.score(X, y)*100,2), '%\n\n')
    
    plot_decision_regions(X, y.ravel(), clf=clf, legend=0, ax=ax, n_jobs=n_jobs).figure.set_size_inches(GEOZ_FIGURE_RESOLUTION * ratio, GEOZ_FIGURE_RESOLUTION)

    return ax



def voronoi_regions_plot(data, lon_col, lat_col, region_col, alpha=None, ax=None, colormap=None, crs=None, show_points=False, mask=None):
    
    '''
    This function creates a Voronoi diagram plot for clustered geographical data. It visualizes the regions
    based on the proximity of data points, with each region colored according to its cluster.

    The main advantage of this method is its ability to show clustering regions for small amount of data as opposed to the other methods.The method also does not rely on machine learning algorithms, making it suitable for quick visualization and
    prototyping of clustering results.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the geographical data and clustering results. Must include columns
        for longitude, latitude, and region/cluster identifier.

    lon_col : str
        The name of the column in 'data' that contains longitude values.

    lat_col : str
        The name of the column in 'data' that contains latitude values.

    region_col : str
        The name of the column in 'data' that contains the region or cluster identifiers.

    alpha : float, optional
        The transparency of the Voronoi regions. If None, full opacity is used.

    ax : matplotlib.axes.Axes, optional
        An existing matplotlib Axes. If provided, the function will draw on this existing axis
        instead of creating a new figure.

    colormap : str or Colormap, optional
        A Colormap instance or registered colormap name. The colormap maps the level values to colors. 
        Defaults to "tab20".
        
    crs : str, optional
        Coordinate Reference System of the Study Area you want to model. if you do not know yours, you can visit the 
        EPSG website (https://spatialreference.org/ref/epsg/) and search for the appropriate authority string for your area, 
        it should be a string like (eg "EPSG:4326") or a WKT string.

    show_points : bool, default True
        Display the points used to create the Decision Boundary. This would allow the user to check the accuracy of the model
        as well as any artifacts or missing clusters, thus alerting the user to enable bazel_cluster function.

    mask : shapely.geometry.Polygon, optional
        A polygon to use as a mask for the Voronoi diagram. If provided, the diagram will be
        clipped to the extent of this shape.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The Axes object with the plotted Voronoi diagram. In IPython environments, this will
        display automatically. The figure can also be accessed using the .figure attribute
        for further manipulation or saving.

    Notes
    -----
    This function uses the Voronoi algorithm from scipy.spatial and matplotlib for plotting.
    The resulting plot divides the space into regions based on the nearest data point, which
    can provide insights into the spatial distribution and boundaries of different clusters,
    in addition to its main task of mapping geographic regions with sparse datasets.

    The color scheme uses the global GEOZ_COLOR_SET variable which is set to uses the 'tab20' colormap. 
    This colormap provides distinct colors for up to 20 different regions. 
    The user can change the global options of GeoZ using the set_geoz_options function.
    
    The figure size is determined by the global GEOZ_FIGURE_RESOLUTION setting. By default it will use 100 inches

    If a mask is provided, the Voronoi diagram will be clipped to the shape of the mask,
    which can be useful for restricting the plot to a specific geographical area.
    
    '''
             
    # Calculate the Figure Dimensions        
    latRange = data[lat_col].max() - data[lat_col].min()
    lonRange = data[lon_col].max() - data[lon_col].min()
    ratio = lonRange/latRange

    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(GEOZ_FIGURE_RESOLUTION * ratio, GEOZ_FIGURE_RESOLUTION))
    else:
        if alpha==None:
            alpha=0.7    
    
    # Create Voronoi diagram
    points = data[[lon_col, lat_col]].values
    vor = Voronoi(points)
    
    # Create a color map
    if colormap is None:
        colormap = GEOZ_COLOR_SET

    # Create a color map
    unique_regions = data[region_col].unique()
    color_map = plt.cm.get_cmap(colormap)  # You can change this to any colormap you prefer
    color_list = [color_map(i/len(unique_regions)) for i in range(len(unique_regions))]
    region_colors = dict(zip(unique_regions, color_list))

    # Prepare patches for all Voronoi regions
    patches = []
    colors = []
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            patches.append(Polygon(polygon))
            region_name = data.loc[r, region_col]
            colors.append(region_colors[region_name])

    # Create a PatchCollection with all polygons
    pc = PatchCollection(patches, facecolors=colors, edgecolors='none', alpha=alpha)

    # Add the collection to the axis
    ax.add_collection(pc)

    # Set the extent of the plot
    ax.set_xlim((data[lon_col].min()-data[lon_col].std()), (data[lon_col].max()+data[lon_col].std()))
    ax.set_ylim((data[lat_col].min()-data[lat_col].std()), (data[lat_col].max()+data[lat_col].std()))
    ax.figure.set_size_inches(GEOZ_FIGURE_RESOLUTION * ratio, GEOZ_FIGURE_RESOLUTION)

    # Apply mask if provided
    if mask is not None:
        # Create a masking polygon
        if np.shape(mask.exterior.coords)[1]==2:
            clip_path = Path(mask.exterior.coords)
        else:
            clip_path = Path(np.array(mask.exterior.coords)[:,:2])
        for geom in mask.interiors:
            clip_path = clip_path.make_compound_path(Path(geom.coords))

        # Create a PathPatch for the mask
        patch = PathPatch(clip_path, facecolor='none', edgecolor='none')
        ax.add_patch(patch)
        pc.set_clip_path(patch)

    #Points for GroundTruth
    if show_points==True:
        #Convert the Points into Geometery file compatible with GeoPandas, then assign it correct coordinated and reproject it to similar projection to the shapefile (Basins)
        gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[lon_col], data[lat_col], crs=crs))
        gdf.plot( marker='o',  color='red', markersize=50, ax=ax)


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

    # Assume first column is latitude and second is longitude 
    long_col = X.columns[0]
    lat_col  = X.columns[1]

    latExtra  = X[lat_col].std()  * extent
    longExtra = X[long_col].std() * extent

    LlatMin  = X[lat_col].min()  - X[lat_col].std()
    LlatMax  = X[lat_col].max()  + X[lat_col].std()
    LlongMin = X[long_col].min() - X[long_col].std()
    LlongMax = X[long_col].max() + X[long_col].std()

    UlatMin  = X[lat_col].min()  - latExtra
    UlatMax  = X[lat_col].max()  + latExtra
    UlongMin = X[long_col].min() - longExtra
    UlongMax = X[long_col].max() + longExtra

    bottom = [[LlatMin, UlatMin], (UlongMin, UlongMax)]
    top    = [[LlatMax, UlatMax], (UlongMin, UlongMax)]
    left   = [[UlatMin, UlatMax], (UlongMin, LlongMin)]
    right  = [[LlatMin, UlatMax], (LlongMax, UlongMax)]
    sides  = [bottom, top, left, right]
    lat    = []
    long   = []

    if n_samples == 'default':
        n_samples = y.value_counts().values[0]

    for side in sides:
        for i in range(math.ceil(n_samples / 4)):
            pointLat  = random.uniform(side[0][0], side[0][1])
            pointLong = random.uniform(side[1][0], side[1][1])
            lat.append(pointLat)
            long.append(pointLong)           

    dataset = pd.DataFrame({long_col: long, lat_col: lat})
    dataset['SynthY'] = np.array(y).max() + 1
    Y = copy.deepcopy(y)
    Y=pd.concat((Y.iloc[:,0], dataset['SynthY']),axis=0,ignore_index= True)
    dataset = dataset[[long_col, lat_col]]
    X = pd.concat((X, dataset), axis=0, ignore_index=True)
    
    return X, Y



__all__ = ["sklearn_plot", "mlx_plot", "convex_hull_plot", "voronoi_regions_plot", "set_geoz_options"]