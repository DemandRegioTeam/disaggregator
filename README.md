# DemandRegio

This project aims at setting up both a database and a python toolkit called `disaggregator` for
- temporal and
- spatial disagregation

of demands of 
- electricity,
- heat and
- natural gas

of the final energy sectors
- private households,
- commerce, trade & services (CTS) and
- industry.


## Installation

Before we really start, please install `conda` through the latest [Anaconda package](https://www.anaconda.com/distribution/) or via [miniconda](https://docs.conda.io/en/latest/miniconda.html). After successfully installing `conda`, open the **Anaconda Powershell Prompt**.  
For experts: You can also open a bash shell (Linux) or command prompt (Windows), but then make sure that your local environment variable `PATH` points to your anaconda installation directory.

Now, in the root folder of the project create an environment to work in that will be called `disaggregator` via

```bash
$ conda env create -f environment.yml
```

which installs all required packages. Then activate the environment

```bash
$ conda activate disaggregator
```

## How to start

Once the environment is activated, you can start a Jupyter Notebook from there

```bash
(disaggregator) $ jupyter notebook
```

As soon as the Jupyter Notebook opens in your browser, click on the `01_Demo_data-and-config.ipynb` file to start with a demonstration:

![Jupyter_View][img_01]

[img_01]: img/jupyter_notebook.png "Jupyter Notebook View"

## Results

![Jupyter_View][img_02]

[img_02]: img/spatial_elc_by_household_sizes.png "Year Electricity Consumption of Private Households"

## How does it work?

For each of the three sectors 'private households', 'commerce, trade & services' and 'industry' the spatial and temporal disaggregation is accomplished through application of various functions. These functions take input data from a database and return the desired output as shwon in the diagram. There are four Demo-Notebooks to present these functions and demonstrate their execution.

![Jupyter_View][img_03]

[img_03]: img/model_overview.png "Schematic diagram of modelling approach"

## Acknowledgements

The development of disaggregator was part of the joint [DemandRegio-Project](https://www.ffe.de/en/topics-and-methods/production-and-market/736-harmonization-and-development-of-methods-for-a-spatial-and-temporal-resolution-of-energy-demands-demandregio) which was carried out by

- Forschungszentrum Jülich GmbH (Simon Burges, Bastian Gillessen, Fabian Gotzens)
- Forschungsstelle für Energiewirtschaft e.V. (Tobias Schmid)
- Technical University of Berlin (Stephan Seim, Paul Verwiebe)

## License

Current version of software written and maintained by Paul A. Verwiebe (TUB)

Original version of software written by Fabian P. Gotzens (FZJ), Paul A. Verwiebe (TUB), Maike Held (TUB), 2019/20.

disaggregator is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html), see [LICENSE](LICENSE) for further information.
