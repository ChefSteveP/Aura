# Aura
A model evaluation pipeline for Llama models. 

In this repo we test the effects of post-training quatization and knowledge distillation on model performance.

# Setup
- Run the setup scripts in `setup_scripts` on a VM. Instructions are posted in `setup_readme.md`.  
- Request Access to Llama Models [Link](https://www.llama.com/llama-downloads/). Once registered, follow download instructions on Hugging Face.
- Run `pip install -r requirements.txt`

# Running the code
All code is ran in `models/main.py`.  There are various flags available to perform various actions such as data preparation, knowledge distillation, quantization, and evaluation. The general sequence of operations would be to run the data preparation in order to obtain a tokenized dataset, run the distillation pipeline to perform knowledge distillation on a model, quantize the knowledge distilled model, and then perform evaluation on a set of models.  

**Data Preparation:**

The following command will tokenize a dataset defined via the `runner.run_data_prep()` function.  This defaults to the Project Gutenberg datset and will store the output as defined in `DATA_DIR` located in `models/constants.py`.
```bash
python models/main.py --dataprep
```

**Knowledge Distillation:**

The following command will perform knowledge distillation on a model where a student and teacher model are provided. In our use case, we had a Llama 1B model serve as the student model and a Llama 3B model serve as the teacher model.  Some hyperparameters are available to tweak in `models/main.py` with the rest located in `runner.run_distill()`.  
```bash
python models/main.py --distill
```

**Quantization:**

The following command will run quantization on a model via [AWQ](https://github.com/casper-hansen/AutoAWQ).  The implementation in `models/main.py` quantizes the standard Llama 1B and Llama 3B models, but any model can be substituted that is valid for quantization.

```bash
python models/main.py --ptq
```

**Evaluation:**

The following command will evaluate a set of models defined in a dictionary where the key is the name of the model that will be displayed in all evaluation metrics, plots, and CSV files, and the value is the provided Hugging Face model.

```bash
python models/main.py --evaluate
```

# Helpful Links
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) - AWQ library for post-training quantization.
- [Project Gutenberg Dataset](https://huggingface.co/datasets/manu/project_gutenberg/) - A dataset containing 60k English novels in addition to ~10k novels available in other languages.
- [Llama Hugging Face Models](https://huggingface.co/meta-llama) - The location of all Hugging Face available Llama models.
- [llama-recipes](https://github.com/meta-llama/llama-recipes) - Community driven scripts and integrations.
