Files Description :
- **environment.yml** is the environment file required to launch MeVisTo. Create your conda environment through the following command :
  *conda env create -f environment.yml*
- **metrics_app_[Version_number].py** is the dash application you need to launch in order to use MeVisTo on your favorite Internet Navigator.
  Once the app is launched, the URL used to access the tool is printed in the console.
- **Metrics.nc**, **TimeSeries.nc** and **Seasonal.nc** are the netcdf files containing the metrics, timeseries, and seasonal cycles used in the app.
- **Params_TUN.nc** is the netcdf file containing the parameter values of your simulation ensemble, as well as the metric values of the associated AMIP experiments.
- **dico_ds_to_sc.py** and **ds_to_sc.py** are files required to properly link the metrics to their respective seasonal cycles.
- The **assets** folder contains files used to custom the dash app’s default appearance.
