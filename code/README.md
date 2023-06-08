This folder contains all the python files required for training an INR model for spinal cord MRI super-resolution. 

### Overview of the files and folders
* `preprocessing/` - folder containing scripts for preprocessing the data. The usage example and preprocessing procedure are described in the readme inside. 
* `config.yaml` - configuration file defining all the necessary inputs, model parameters for training an INR. 
* `dataset_utils.py` - utility file containing the functions used during data loading
* `dataset.py` - code for fetching the subjects, converting image data to coordinates, and storing input metadata
* `main.py` - main file for loading the dataset, instantiating and training the model, generating the outputs during inference
* `model.py` - file for defining the neural network model
* `utils.py` - general utility file for various helper functions

### Training procedure 

The procedure for training is extremely simple. The inputs along with network parameters must be defined in the configuration file `config.yaml`. For training the model, run the following command: 

```bash
python main.py --config config.yaml --logging 
```

Note that this assumes that the user has a `wandb` account as the training curves and reconstruction metrics are logged onto `wandb` in real-time. 