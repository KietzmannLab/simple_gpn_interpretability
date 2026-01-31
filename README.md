# Prediction RNN Interpretability 
Official codebase of the "Mechanisms of sequential world modeling" project  
This repository is geared towards making all the code and analyses from the paper possible.  

## Codebase map  
1. The figures included in the paper can be plotted using [paper_plots](paper_plots.ipynb)  
2. The training of the network can be done with [train_model](setup/train_model.py)  
3. Scripts to generate the token sequences can be found under [sequence_maker](setup/sequence_maker.py)  
4. Scripts to train the SVMs for decoding analyses can be done using [run_decoders](decoding_analysis/run_decoders.ipynb)  

## Requirements  

### Environment setup  
The code has been tested with:  
- Python 3.13.5 + PyTorch 2.10.0 (CPU) on macOS (Apple M1)  
- Python 3.13.5 + PyTorch 2.10.0 on a Linux HPC cluster  

Clone this repository (or download it directly):  

```bash  
git clone https://github.com/KietzmannLab/simple_gpn_interpretability.git  
cd simple_gpn_interpretability  
```

Setup your enviroment as:

```bash
conda env create -f environment.yml
conda activate gpn

```  

### Pretrained networks and analysis data  
1. The model, SVM results, and analysis sequences used in the paper are available under [paper_data](paper_data).  

