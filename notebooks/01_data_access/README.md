# Data access

Notebooks covering data discovery and remote access.

- `cmip7_data_availability_and_loading.ipynb`: searches ESGF for each CMIP7 sea ice variable across all experiments, builds a variable × experiment table of the models that published it (saved to `results/summaries/cmip7_availability.csv`), then downloads and plots monthly sea ice concentration for one model.
