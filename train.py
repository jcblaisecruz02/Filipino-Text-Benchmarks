import argparse
from utils.training import run_finetuning

def main():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--pretrained', type=str, required=True, help='Name of (or path to) the pretrained model.')
    parser.add_argument('--checkpoint', type=str, help='Name of output model/checkpoint to evaluate on test data.')
    parser.add_argument('--train_data', type=str, help='Path to the training data.')
    parser.add_argument('--valid_data', type=str, help='Path to the validation data.')
    parser.add_argument('--test_data', type=str, help='Path to the testing data.')
    parser.add_argument('--data_pct', type=float, default=1.0, help='Percentage of training data to train on. Reduce to simulate low-resource settings.')
    parser.add_argument('--text_columns', type=str, default='text', help='Column name(s) of the features. Comma-separated for entailment tasks')
    parser.add_argument('--label_columns', type=str, default='label', help='Column name(s) of the labels to predict. Comma-separated for multilabels.')
    parser.add_argument('--retokenize_data', action='store_true', help='Retokenize and generate data again.')
    parser.add_argument('--msl', type=int, default=128, help='Maximum sequence length.')
    
    # Training parameters
    parser.add_argument('--do_train', action='store_true', help='Finetune a model.')
    parser.add_argument('--do_eval', action='store_true', help='Evaluate a model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--accumulation', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--add_token', action='append', help='Additional special tokens. Will not be split/tokenized separately.')
    parser.add_argument('--random_init', action='store_true', help='Use a randomly-initialized model based off of pretrained configurations.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam epsilon parameter.')
    parser.add_argument('--use_scheduler', action='store_true', help='Use a linearly-decaying scheduler.')
    parser.add_argument('--warmup_pct', type=float, default=0.1, help='Percentage of the training steps to warmup learning rate.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use the GPU.')
    parser.add_argument('--dont_save', action='store_true', help='Do not save the finetuned checkpoint.')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Log experiment results in Weights and Biases.')
    parser.add_argument('--wandb_project_name', type=str, help='Name of project in Weight and Biases.')
    parser.add_argument('--wandb_username', type=str, help='Username in Weight and Biases.')
    
    args = parser.parse_args()
    
    # Log the configuration
    print('=' * 50, '\nCONFIGURATION', '\n' + '=' * 50)
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))

    # Run finetuning
    metrics = run_finetuning(args)

    # Weights and Biases
    if args.use_wandb:
        print('\n' + '=' * 50, '\nWEIGHTS AND BIASES LOGGING', '\n' + '=' * 50)
        import wandb
        wandb.init(entity=args.wandb_username, project=args.wandb_project_name, reinit=True)
        config = wandb.config.update(args)

        wandb.log({"Train Loss": metrics[0], 
                   "Train Accuracy": metrics[1],
                   "Validation Loss": metrics[2], 
                   "Valiation Accuracy": metrics[3],
                   "Test Loss": metrics[4], 
                   "Test Accuracy": metrics[5],})


if __name__ == '__main__':
    main()
