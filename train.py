import argparse
from utils.training import run_finetuning
from utils.data import str2bool

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
    parser.add_argument('--retokenize_data', type=str2bool, default='false', help='Force the script to generate data cache again.')
    parser.add_argument('--save_cache', type=str2bool, default='true', help='Save the data cache.')
    parser.add_argument('--msl', type=int, default=128, help='Maximum sequence length.')
    
    # Training parameters
    parser.add_argument('--do_train', type=str2bool, default='true', help='Finetune the model.')
    parser.add_argument('--do_eval', type=str2bool, default='true', help='Evaluate the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Which optimizer to use.')
    parser.add_argument('--adam_b1', type=float, default=0.9, help='Adam Beta1.')
    parser.add_argument('--adam_b2', type=float, default=0.999, help='Adam Beta2.')
    parser.add_argument('--accumulation', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--add_token', type=str, default='', help='Additional special tokens.')
    parser.add_argument('--random_init', type=str2bool, default='false', help='Randomly initialize the model')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam epsilon parameter.')
    parser.add_argument('--use_scheduler', type=str2bool, default='true', help='User a scheduler.')
    parser.add_argument('--warmup_pct', type=float, default=0.1, help='Percentage of the training steps to warmup learning rate.')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--no_cuda', type=str2bool, default='false', help='Do not use GPU.')
    parser.add_argument('--dont_save', type=str2bool, default='false', help='Do not save finetuned model.')
    parser.add_argument('--fp16', type=str2bool, default='false', help='Use FP16 Training via APEX.')
    parser.add_argument('--opt_level', type=str, default='O1', help='Opt level for mixed precision training.')
    
    # Logging
    parser.add_argument('--use_wandb', type=str2bool, default='false', help='Use wandb logging.')
    parser.add_argument('--wandb_project_name', type=str, help='Name of project in Weight and Biases.')
    parser.add_argument('--wandb_username', type=str, help='Username in Weight and Biases.')
    
    args = parser.parse_args()
    
    # Log the configuration
    print('=' * 50, '\nCONFIGURATION', '\n' + '=' * 50)
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))

    # Setup weights and biases
    if args.use_wandb:
        import wandb
        wandb.init(entity=args.wandb_username, project=args.wandb_project_name, reinit=True)
        config = wandb.config.update(args)

    # Run finetuning
    metrics = run_finetuning(args)

    # Logging
    if args.use_wandb:
        print('\n' + '=' * 50, '\nWEIGHTS AND BIASES LOGGING', '\n' + '=' * 50)
        wandb.log({"Train Loss": metrics[0], 
                   "Train Accuracy": metrics[1],
                   "Validation Loss": metrics[2], 
                   "Validation Accuracy": metrics[3],
                   "Test Loss": metrics[4], 
                   "Test Accuracy": metrics[5],})


if __name__ == '__main__':
    main()
