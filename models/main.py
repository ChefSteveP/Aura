# main.py
from data_preparation import DataPreparation
from model_trainer import ModelTrainer
from model_pruner import ModelPruner
from model_quantizer import ModelQuantizer
from model_evaluator import ModelEvaluator


def main():
    # general
    model_name = "meta-llama/Llama-3.2-1B"
    huggingface_token = "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"

    # data prep vars
    saved_data_path = "~/storage/data/project_gutenberg"
    download_path = "manu/project_gutenberg"
    split = "en"
    tokenized_field_name = "text"

    # training vars
    model_results_path = "results/llama_3-2_1B"
    num_labels = 2
    epochs = 3
    batch_size = 8
    learning_rate = 2e-5
    wandb_project = "aura-ai"

    # if data not cached: download, tokenize, and save data locally
    # else, load from cache
    data_preparation = DataPreparation(
        model_name=model_name,
        huggingface_token=huggingface_token,
        saved_data_path=saved_data_path,
        download_path=download_path,
        split=split,
        tokenized_field_name=tokenized_field_name,
    )
    dataset = data_preparation.prepare_data()

    # train model
    model_trainer = ModelTrainer(
        model_name=model_name,
        saved_data_path=saved_data_path,
        model_results_path=model_results_path,
        num_labels=num_labels,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        wandb_project=wandb_project,
    )
    model_trainer.train_model()

    # # Step 3: Model Pruning
    # model_pruner = ModelPruner()
    # model_pruner.prune_model(amount=0.3)
    # model_pruner.save_model()

    # Step 4: Model Quantization
    #PTQ
    model_quantizer = ModelQuantizer()
    model_quantizer.ptq()
    # model_quantizer.quantize_model()
    model_quantizer.save_model()
    # #QAT
    # model_trainer.qat()

    # # Step 5: Model Evaluation
    # model_evaluator = ModelEvaluator()
    # model_evaluator.evaluate()


if __name__ == "__main__":
    main()
