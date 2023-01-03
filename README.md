![PyPI](https://img.shields.io/pypi/v/geoz) ![PyPI - Downloads](https://img.shields.io/pypi/dm/geoz?color=dark%20green)
# Geographic Decision Zones (GeoZ)


GeoZ is a Python library integrating several machine learning modules to create Geographic Maps based on the output of 
Unsupervised Machine Learning techniques. The library is geared mainly toward delineating the output from Clustering 
algorithms, but it can be used for other Machine Learning algorithms. GeoZ is distributed under the 3-Clause BSD license.

## Installation

**To install GeoZ using `pip`:**
```bash
pip install geoz
```
## Usage Details

The library is still in its inital stage. As such, the user will have to provide the data in a certain format as the library is working with a fixed structure and wont fix or tolerate any deviation from the expected format.

### Dataset shape and format Example
The data provided needs to have two variables, one containing the latitude and longitude (eg. latlong) and another variable that contains the predicted classes of the the points (eg. y_pred). please check the below table for illustration:

| LATITUDE 	| LONGITUDE 	| y_pred 	|
|:--------:	|:---------:	|:------:	|
|    30    	|    -104   	|    2   	|
|    32    	|    -103   	|    1   	|
|    35    	|    -105   	|    2   	|
|    33    	|    -104   	|    2   	|
|    35    	|    -102   	|    3   	|

Please make sure to write (LATITDE, LONGITUDE) in CAPITAL LETTER, otherwise the algorithm will fail.

### Code Example

In this example, we import geoz and then use an already defined variable 'dataset' that contains our above table, the variable can contain only the latitude and longitude, but we will need to use another variable y_pred to store the class predictions and use it in the functions calling.

```python
import geoz

dataset=dataset                           # This is supposed to be the dataset that you have, it must contain the Latitude and the longitude as well as the class information

map1 = geoz.convex_hull_plot(dataset[['LATITDE','LONGITUDE']], dataset[['y_pred']])            # This Function will return a Convex Hull map of the classes

map2 = geoz.sklearn_plot(dataset[['LATITDE','LONGITUDE']], dataset[['y_pred']])                # This Function will return a map drawn using Scikit-Learn "DecisionBoundaryDisplay"

map3 = geoz.mlx_plot(dataset[['LATITDE','LONGITUDE']], dataset[['y_pred']])                    # This Function will return a map drawn using MLextend  "decision_regions"

```

For any other infromation or the functions other parameters, please check the functions DocStrings as they contain more details and information.

## License information

See the file ([LICENSE](https://github.com/Ne-oL/geoz/blob/main/LICENSE)) for information on the terms & conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

## Contact

You can ask me any questions via my Twitter Account [Ne-oL](https://twitter.com/Ne_oL). and in case you encountered any bugs, please create an issue in [GitHub's issue tracker](https://github.com/Ne-oL/geoz/issues) and I will try my best to address it as soon as possible. 

## Publication
Publication is under Process if you use this library in the mean time, please cite the Github repositry


Created By Khalid ElHaj, PhD Fellow

Geosciences Department @ UAE University
