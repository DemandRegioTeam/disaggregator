# DemandRegio and Disaggregator

## About
The energy transition and increased supply of electricity from intermitting renewable energy sources leads to significant changes in the energy system. An analysis now requires models with a high temporal and spatial resolution. While extensive high-resolution models and data are already available in transparent form for electricity generation, this does not the case for the energy demand side.

The aim of the DemandRegio research project was to develop a uniform and transparent method for analysing the temporal and spatial resolution of electricity and gas demand. The method was implemented as a modeling tool and made available together with harmonized input and result data. 

The tool `disaggregator` helps you to build a database, disaggregate (temporal and spatial disaggregation) and analyse data for the demands of **electricity**, **heat** and **natural gas** of the final energy sectors **private households**, **commerce, trade & services (CTS)** and **industry**.

A detailed explanation of the research project, the background of the research project and a detailed description of current and potential application areas can be found in the project report:

	Gotzens, Fabian, Bastian Gillessen, Simon Burges, Wilfried Hennings,
	Joachim Müller-Kirchenbauer, Stephan Seim, Paul Verwiebe, 
	Schmid Tobias, Fabian Jetter, und Timo Limmer. 
	"DemandRegio - Harmonisierung und Entwicklung von Verfahren 
	zur regionalen und zeitlichen Auflösung von Energienachfragen:
	Abschlussbericht“, 2020. https://doi.org/10.34805/FFE-119-20.
	
## Installation and get started using disaggregator

The current version of disaggregator works with Conda and Jupyter Notebooks.

Please install `conda` through the latest [Anaconda package](https://www.anaconda.com/distribution/) or via [miniconda](https://docs.conda.io/en/latest/miniconda.html). After successfully installing `conda`, open the **Anaconda Powershell Prompt**.
(For experts: You can also open a bash shell (Linux) or command prompt (Windows), but then make sure that your local environment variable `PATH` points to your anaconda installation directory.)

Now, in the root folder of the project create an environment to work in that will be called `disaggregator` via

```bash
$ conda env create -f environment.yml
```

which installs all required packages. Then activate the environment

```bash
$ conda activate disaggregator
```

You might need to download additional python packages for running `disaggregator`. These include:
- Geopandas (https://pypi.org/project/geopandas/)
- Holidays: (https://pypi.org/project/holidays/)
- Matplotlib (https://pypi.org/project/matplotlib/)
- Pandas (https://pypi.org/project/pandas/)
- Xarray (https://pypi.org/project/xarray/)

A full list of packages used can be found in the file `environment.yml` <environment.yml>.


## Tutorials for using disaggregator

Once the environment is installed, you can start a Jupyter Notebook from there

```bash
(disaggregator) $ jupyter notebook
```

There are five tutorials and demonstration examples that help you to better understand `disaggregator` and its applications. The files can be found in the folder `docs/tutorials`.

1. [Data and configuration](tutorials/01_Demo_data-and-config.ipynb)
2. [Households: spatial disaggregation](tutorials/02_Demo_households_spatial_disaggregations.ipynb)
3. [Households: temporal disaggregation for power and gas](tutorials/03_Demo_households_temporal_disaggregations_power_and_gas.ipynb)
4. [CTS / Industry disaggregation](tutorials/04_Demo_CTS_Industry_disaggregation.ipynb)
5. [Accessing geographical data](tutorials/05_Demo_accessing_geographical_data.ipynb)

Click on the [`01_Demo_data-and-config.ipynb`](tutorials/01_Demo_data-and-config.ipynb) file to start with a demonstration:

![Jupyter_View][img_01]

[img_01]: docs/_static//jupyter_notebook.png "Jupyter Notebook View"

## Results

![Jupyter_View][img_02]

[img_02]: docs/_static/spatial_elc_by_household_sizes.png "Year Electricity Consumption of Private Households"


## How does it work?

For each of the three sectors 'private households', 'commerce, trade & services' and 'industry' the spatial and temporal disaggregation is accomplished through application of various functions. These functions take input data from a database and return the desired output as shwon in the diagram. There are four Demo-Notebooks to present these functions and demonstrate their execution.

![Jupyter_View][img_03]

[img_03]: docs/_static//model_overview.png "Schematic diagram of modelling approach"

## Acknowledgements

The development of disaggregator was part of the joint [DemandRegio-Project](https://www.ffe.de/en/topics-and-methods/production-and-market/736-harmonization-and-development-of-methods-for-a-spatial-and-temporal-resolution-of-energy-demands-demandregio) which was carried out by

- Forschungszentrum Jülich GmbH (Simon Burges, Bastian Gillessen, Fabian Gotzens)
- Forschungsstelle für Energiewirtschaft e.V. (Tobias Schmid) <http://www.ffe.de>
- Technical University of Berlin (Stephan Seim, Paul Verwiebe) <http://www.er.tu-berlin.de>

If `disaggregator` has been significant in a project that leads to a publication, please acknowledge that by citing the project report (DOI: 10.34805/FFE-119-20) linked above.


## License
The current version of software was written and is maintained by Paul A. Verwiebe (TUB).

The original version of software has written by Fabian P. Gotzens (FZJ), Paul A. Verwiebe (TUB), Maike Held (TUB), 2019/20.

disaggregator is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html), see [LICENSE](LICENSE) for further information.


## Contributing to disaggregator 🎁
Disaggregator is designed as an open sources tool and welcomes contributions. You can contribute to the disaggregator software on GitHub directly.

If you have a question that isn't answered in the tutorials or the documentation please `open an issue on GitHub <https://github.com/DemandRegioTeam/disaggregator/issues>`_, if there isn't one there already.
