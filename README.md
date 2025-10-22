# dswt_detector
Detection of dense shelf water transport and cascades

This code detects dense shelf water transport (DSWT) on continental shelves in 3D ocean model output. It does this by:
1. Generating cross-shelf transects roughly perpendicular to depth contours using bathymetry data from the ocean model.
2. Determining if DSWT is happening in a transect at a particular time.
3. Calculating the associated cross-shelf transport if DSWT is occurring.

The code currently only works for ocean model data in a ROMS output format and it assumes that each netcdf file contains daily data.

DSWT is (among others) detected using a threshold for the vertical density gradient. An appropriate value for this threshold needs to be determined depending on the location and the ocean model used. The value can be set in a config file (see below).

## Getting started
I recommend creating a `dirs.json` file in the `input` folder (though the main script currently runs using directory information in the `input/example_dirs.json` file). The function `get_dir_from_file` in `tools.files` by default looks for directory information in the `input/dirs.json` file. This should contain paths to ocean model data and a directory where you want plots to end up, as per the following example:
```
{
    "your_model": "path/to/ocean/model/data/",
    "plots": "path/to/plot/output/"
}
```
Output files from the DSWT detection are stored in the `output` folder by default.

The `input/configs/main_config.toml` file contains config settings for the DSWT detection. Many of these are default settings that can be kept the same, but a `minimum_drhodz` value needs to be set for each model specifically. This is the threshold value that the vertical density gradient needs to exceed to qualify as DSWT. Example values for some models are already given in the `main_config.toml` file. Add your own model as follows:
```
[your_model]
minimum_drhodz = appropriate_float_value
```

Once you have appropriate ocean model data files, paths, and config settings you can run the `main_dswt_detector.py` script after making relevant changes to the "User input" section. The changes you need to make here should be relatively self-evident.
The `main_dswt_detector.py` script should also run without any changes at all using the test data contained in this repository as well; just run this script if you want to make sure everything is working.

## DSWT detection (running `main_dswt_detector.py`)
Running `main_dswt_detector.py` will go through several steps (skipping steps where it can if they have already been done before).

### Transect creation
Cross-shelf transects that are approximately perpendicular to depth contours are created first, and are saved to the `input/transects/` folder. See `input/transects/test_transects.json` for an example based on the test data (in `tests/data/`).
Depth contours specified in `main_config.toml` under `transect_contours` are used to generate the transects. The first depth contour in `transect_contours` is used as the starting contour (10 m by default).

The automatically generated transects will probably not be perfect, so the script gives you the option to remove transects manually by loading a GUI and allowing you to click on the transects you want to remove. Remove any transects that cross others, as this will give problems when calculating cross-shelf transport. The figure below shows an example with transects (in red) before and after manual removal. This GUI is run by the `main_dswt_detector.py` script if there is no transects file available for your model yet. The GUI is run by the `interactive_transect_removal` function in `guis.transect_removal`.

<img width="516" alt="transect_removal" src="https://github.com/user-attachments/assets/0c65f44e-fd6a-4088-9b1c-eaa0ab322531" />

After manually removing transects, it is also possible to add transects back in specific areas and starting from a different depth contour. This can be useful around islands, although it does create some new problems down the line (for example, if you start the transect quite deep, DSWT may be wrongly detected; I have tried to solve this in a processing step but this is not ideal). To add transects back in, another GUI is run which allows you to create a polygon (by clicking to create points) around the area where you want new transects and then allows you to click on the starting contour line for transects in this polygon. The figure below shows an example of a polygon created in the GUI where transects will be added back in. After this, the `main_dswt_detector` script will again run the GUI that allows you to remove transects, in case any of the new transects need to be removed. The GUI to add transects is run by the `interactive_transect_addition` function in `guis.transect_addition`. The `main_dswt_detector` script also saves the names of the transects that have been added in this step in a `input/transects/<your_model>_transects_islands.csv` file, which is used for processing later.

<img width="468" alt="transect_addition-2" src="https://github.com/user-attachments/assets/67f6d817-e0f9-4dae-b723-1d3e3c14e59e" />

### Config settings
As mentioned above, the `minimum_drhodz` config setting needs to be determined manually and is specific to your region of interest and the ocean model you are using. A GUI run by `interactive_transect_time_cycling_plot` in `guis.check_dswt_config` can help you do this. The `main_dswt_detector` script will also run this GUI if you indicate you want to check your config settings. You will be asked for which day to load ocean model data and a map of both the bottom and surface density will be shown, along with the transects. You can click on a transect and the density, temperature, salinity, and vertical density gradient will be shown as well. You can cycle through in time using keyboard arrows. You can use this GUI to find a time and location when DSWT occurs, which can help you determine which value to set for `minimum_drhodz` for your purposes. The figure below shows an example of the GUI.

<img width="578" alt="settings_helper" src="https://github.com/user-attachments/assets/79ec32fc-30f2-4846-8710-4513ca95a06d" />

## Performance check
After confirming your config settings, the `main_dswt_detector` script will give you the option to run a performance check. For this, you will be shown the density, temperature, and salinity along random transects during random times and you will be asked to visually determine whether there is DSWT occurring or not, which is then compared to what the algorithm determined. The results of this will be saved in a csv file in the `performance_tests/output/` folder. You will be told how well the algorithm agree with your visual determination of DSWT. You can also run this using the `manual_random_checks.py` and `rate_performance.py` scripts in the `performance_tests` folder.

## DSWT detection and cross-shelf transport calculation
Next, the DSWT detection and cross-shelf transport calculation alrogithm will be run (functions for this are in the scripts in the `dswt` folder). This is run for each daily ocean model output file and for each transect. Outcomes are written to csv files (separate csv files for each year) and by default are stored in the `output` folder. To avoid unnecessary repetitions, daily values are appended to the csv file and are skipped if they already exist in the file.

## Processing
After DSWT detection and cross-shelf transport calculation, the output is processed. There are several known cases in which DSWT is wrongly detected. In the processing step, these faulty detections are removed (saved in new output files by default in the `output/processed/` folder). Ideally, these would be caught and not classified as DSWT by the detection algorithm. This is still a work in progress to be improved.

## Plotting
Two example figures are made using the DSWT detection output: a timeseries showing the monthly mean DSWT occurrence and associated cross-shelf transport and a map showing the overall mean cross-shelf DSWT in each location. Plotting functions can be found in `main_plots.py` and `read_dswt_output.py` in the `readers` folder contains several useful functions to read and convert the output data into a useful format. Example plots for the small test included in the repo are shown in the `plots` folder. The figures below are examples of the timeseries and map plot for a full year.

<img width="433" alt="cwa_timeseries" src="https://github.com/user-attachments/assets/41e52cd6-cb6d-4dc2-b5af-ce18dc82fef1" />

<img width="309" alt="cwa_map" src="https://github.com/user-attachments/assets/99f52705-adb7-43da-b0b1-de93d61a1630" />
