import argparse
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from tqdm import tqdm

def tokenize(t, tokenizer, msl): 
    return tokenizer.encode(t, pad_to_max_length=True, max_length=msl)

def accuracy(out, y): 
    # Columnwise mean accurarcy for multilabel classification
    if len(y.shape) > 1:
        return ((out.sigmoid() > 0.5).long() == y).float().mean(0).mean()
    # Standard accuracy
    return (out.argmax(1) == y).sum().item() / len(y)

def train(model, criterion, optimizer, train_loader, scheduler=None, accumulation=1, device=None):
    model.train()
    train_loss, train_acc = 0, 0
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        out = model(x)[0]

        loss = criterion(out, y)
        loss.backward() 

        if (i + 1) % accumulation == 0: 
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()

        train_loss += loss.item()
        train_acc += accuracy(out, y)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc

def evaluate(model, criterion, valid_loader, device=None):
    model.eval()
    valid_loss, valid_acc = 0, 0
    for x, y in tqdm(valid_loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            out = model(x)[0]

            loss = criterion(out, y)

        valid_loss += loss.item()
        valid_acc += accuracy(out, y)
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)

    return valid_loss, valid_acc

def finetune(args):
    # Wrap the finetuning function in an objective to support
    # Hyperparameter search using Optuna
    def objective(trial):
        # Suggest hyperparameters
        if args.optimize_learning_rate: 
                lr = trial.suggest_loguniform('learning_rate', args.opt_lr_lowerbound, args.opt_lr_upperbound)
        else: 
            lr = args.learning_rate
        if args.optimize_weight_decay: 
            wd = trial.suggest_loguniform('weight_decay', args.opt_wd_lowerbound, args.opt_wd_upperbound)
        else: 
            wd = args.weight_decay
        if args.optimize_seed: 
            seed = trial.suggest_int('seed', args.opt_seed_lowerbound, args.opt_seed_upperbound)
        else: 
            seed = args.seed

        if args.use_wandb:
            import wandb
            wandb.init(entity=args.wandb_username ,project=args.wandb_project_name, reinit=True if args.optimize_hyperparameters else False)
            config = wandb.config
            config.seed = seed
            config.batch_size = args.batch_size
            config.accumulation = args.accumulation
            config.pretrained_model = args.pretrained
            config.data_pct = args.data_pct
            config.use_scheduler = args.use_scheduler
            config.warmup_pct = args.warmup_pct
            config.lowercase = args.lowercase
            config.weight_decay = wd
            config.learning_rate = lr
            config.random_init = args.random_init
            wandb.save(args.study_name)

        # Preliminaries
        print("\n===== BEGIN FINETUNING =====\n")
        print("Using random seed", seed)
        torch.manual_seed(seed)
        
        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained, model_max_length=args.msl, do_lower_case=args.lowercase)
        add_token = {'additional_special_tokens': args.add_token}
        added = tokenizer.add_special_tokens(add_token)
        if added > 0: print("Addded {} special tokens:".format(added), args.add_token)

        # Set the label columns
        columns = (args.label_column).split(',')
        num_labels = len(columns)
        if num_labels == 1: columns = columns[0]

        # Training proper
        if args.do_train:
            # Load datasets
            print("Loading and tokenizing dataset")
            print("Using {:.2f} of the training set.".format(args.data_pct))
            df = pd.read_csv(args.train_data).sample(frac=args.data_pct, random_state=seed)
            X_train = torch.tensor(np.array([tokenize(t, tokenizer, msl=args.msl) for t in tqdm(df[args.text_column])]))
            y_train = torch.tensor(df[columns].values)
            if num_labels > 1: y_train = y_train.float()
            
            df = pd.read_csv(args.valid_data).sample(frac=1.0, random_state=seed)
            X_valid = torch.tensor(np.array([tokenize(t, tokenizer, msl=args.msl) for t in tqdm(df[args.text_column])]))
            y_valid = torch.tensor(df[columns].values)
            if num_labels > 1: y_valid = y_valid.float()

            # Dataloaders
            train_set = torch.utils.data.TensorDataset(X_train, y_train)
            valid_set = torch.utils.data.TensorDataset(X_valid, y_valid)
            train_sampler = torch.utils.data.RandomSampler(train_set)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
            valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

            # Model setup
            device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
            config = AutoConfig.from_pretrained(args.pretrained, num_labels=2 if num_labels == 1 else num_labels)
            if args.random_init:
                print("Initializing new randomly-initialized model from configuration")
                model = AutoModelForSequenceClassification.from_config(config)
            else:
                print("Loading from pretrained checkpoint")
                model = AutoModelForSequenceClassification.from_pretrained(args.pretrained, config=config)
            _ = model.resize_token_embeddings(len(tokenizer))
            model = model.to(device)
            print("Model has {:,} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

            # Initialize loss, optimizer, and scheduler
            criterion = torch.nn.CrossEntropyLoss() if num_labels == 1 else torch.nn.BCEWithLogitsLoss()
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                                            "weight_decay": wd}, 
                                            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                                            "weight_decay": 0.0}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
            optimizer.zero_grad()

            if args.use_scheduler:
                steps = len(train_loader) * args.epochs // args.accumulation
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps * args.warmup_pct), num_training_steps=steps)
            else: scheduler = None

            print("Using learning rate {:.4E} and weight decay {:.4E}".format(lr, wd), end='')
            print(" with scheduler using warmup pct {}".format(args.warmup_pct)) if args.use_scheduler else print("")

            if args.use_wandb:
                wandb.watch(model, log="all")

            # Train
            print("\nBegin training.")
            for e in range(1, args.epochs + 1):
                train_loss, train_acc = train(model, criterion, optimizer, train_loader, scheduler=scheduler, accumulation=args.accumulation, device=device)
                valid_loss, valid_acc = evaluate(model, criterion, valid_loader, device=device)
                
                # Pruning (hyperparam opt)
                if args.optimize_hyperparameters and args.opt_use_pruning:
                    trial.report(valid_acc, e)
                    if trial.should_prune(): raise optuna.TrialPruned()

                print("Epoch {:3} | Train Loss {:.4f} | Train Acc {:.4f} | Valid Loss {:.4f} | Valid Acc {:.4f}".format(e, train_loss, train_acc, valid_loss, valid_acc))

                if args.use_wandb:
                    wandb.log({
                        "Train Loss": train_loss,
                        "Train Accuracy": train_acc,
                        "Validation Loss": valid_loss,
                        "Validation Accuracy": valid_acc})

            # Save the checkpoint
            if not args.dont_save:
                with open(args.checkpoint, 'wb') as f:
                    torch.save(model.state_dict(), f)

            print("\n===== FINISHED FINETUNING =====\n")

        # Testing proper
        if args.do_eval:
            # Load the dataset
            print("Loading and tokenizing test dataset")
            df = pd.read_csv(args.test_data).sample(frac=1.0, random_state=seed)
            X_test = torch.tensor(np.array([tokenize(t, tokenizer, msl=args.msl) for t in tqdm(df[args.text_column])]))
            y_test = torch.tensor(df[columns].values)
            if num_labels > 1: y_test = y_test.float()

            # Dataloaders
            test_set = torch.utils.data.TensorDataset(X_test, y_test)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

            # Produce the model
            device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
            config = AutoConfig.from_pretrained(args.pretrained, num_labels=2 if num_labels == 1 else num_labels)
            model = AutoModelForSequenceClassification.from_config(config)
            _ = model.resize_token_embeddings(len(tokenizer))
            model = model.to(device)

            # Load checkpoints
            print("Loading finetuned checkpoint")
            with open(args.checkpoint, 'rb') as f:
                model.load_state_dict(torch.load(f))
            criterion = torch.nn.CrossEntropyLoss() if num_labels == 1 else torch.nn.BCEWithLogitsLoss()
            
            # Test
            test_loss, test_acc = evaluate(model, criterion, test_loader, device=device)
            print("Test Loss {:.4f} | Test Accuracy {:.4f}".format(test_loss, test_acc))

            if args.use_wandb:
                wandb.log({"Test Loss": test_loss, "Test Accuracy": test_acc})

            # Prevent errors from not training
            if not args.do_train: valid_acc = 0

        # Return validation accuracy as hyperparam tuning objective
        return valid_acc

    # Run finetuning
    # Use hyperparameter search
    if args.optimize_hyperparameters:
        assert args.do_train == True
        import optuna

        study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), 
                                    direction='maximize', 
                                    study_name=args.study_name, 
                                    storage='sqlite:///' + args.study_name, 
                                    load_if_exists=True)
        study.optimize(objective, n_trials=args.opt_n_trials)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial
        print("  Validation Accuracy : {:.4f}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            if key in ['learning_rate', 'weight_decay']: print("    {}: {:.4E}".format(key, value))
            else: print("    {}: {}".format(key, value))
    
    # Run standard finetuning without hyperparameter search
    else:
        _ = objective(None)

def main():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--pretrained', type=str, required=True, help='Name of (or path to) the pretrained model.')
    parser.add_argument('--checkpoint', type=str, help='Name of output model/checkpoint to evaluate on test data.')
    parser.add_argument('--train_data', type=str, help='Path to the training data.')
    parser.add_argument('--valid_data', type=str, help='Path to the validation data.')
    parser.add_argument('--test_data', type=str, help='Path to the testing data.')
    parser.add_argument('--data_pct', type=float, default=1.0, help='Percentage of training data to train on. Reduce to simulate low-resource settings.')
    parser.add_argument('--text_column', type=str, default='text', help='Column name of the features.')
    parser.add_argument('--label_column', type=str, default='label', help='Column name(s) of the labels to predict. Comma-separated for multilabels.')
    parser.add_argument('--lowercase', action='store_true', help='Set when using an uncased model.')
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
    parser.add_argument('--use_wandb', action='store_true', help='Log experiment results in Weights and Biases.')
    parser.add_argument('--wandb_project_name', type=str, help='Name of project in Weight and Biases.')
    parser.add_argument('--wandb_username', type=str, help='Username in Weight and Biases.')
    
    # Hyperparameter optimization arguments
    parser.add_argument('--optimize_hyperparameters', action='store_true', help='Toggle to use hyperparameter optimization.')
    parser.add_argument('--opt_use_pruning', action='store_true', help='Prune unpromising runs.')
    parser.add_argument('--opt_n_trials', type=int, default=100, help='Number of trials to run hyperparameter optimization.')
    parser.add_argument('--study_name', type=str, default='my_study', help='Name of generated output database for hyperparameter optimization.')
    parser.add_argument('--optimize_seed', action='store_true', help='Toggle to optimize the seed.')
    parser.add_argument('--opt_seed_lowerbound', type=int, default=1, help='Lower bound for seed optimization.')
    parser.add_argument('--opt_seed_upperbound', type=int, default=99, help='Upper bound for seed optimization.')
    parser.add_argument('--optimize_learning_rate', action='store_true', help='Toggle to optimize the learning rate.')
    parser.add_argument('--opt_lr_lowerbound', type=float, default=1e-5, help='Lower bound for learning rate optimization.')
    parser.add_argument('--opt_lr_upperbound', type=float, default=5e-4, help='Upper bound for learning rate optimization.')
    parser.add_argument('--optimize_weight_decay', action='store_true', help='Toggle to optimize weight decay.')
    parser.add_argument('--opt_wd_lowerbound', type=float, default=1e-8, help='Lower bound for weight decay optimization.')
    parser.add_argument('--opt_wd_upperbound', type=float, default=1e-3, help='Upper bound for weight decay optimization.')
    args = parser.parse_args()
    print(args)

    # Run finetuning
    finetune(args)

if __name__ == '__main__':
    main()
