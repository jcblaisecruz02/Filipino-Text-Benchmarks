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
        # Preliminaries
        print("\n===== BEGIN FINETUNING =====\n")
        if args.optimize_seed: 
            seed = trial.suggest_int('seed', args.opt_seed_lowerbound, args.opt_seed_upperbound)
        else: 
            seed = args.seed
        print("Using random seed", seed)
        torch.manual_seed(seed)

        # Initialize Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained, model_max_length=args.msl, do_lower_case=args.lowercase)
        add_token = {'additional_special_tokens': args.add_token}
        added = tokenizer.add_special_tokens(add_token)
        if added > 0: print("Addded {} special tokens:".format(added), args.add_token)

        # Load datasets
        print("Loading and tokenizing dataset")
        df = pd.read_csv(args.train_data).sample(frac=args.data_pct, random_state=args.seed)
        X_train = torch.tensor(np.array([tokenize(t, tokenizer, msl=args.msl) for t in tqdm(df[args.text_column])]))
        y_train = torch.tensor(np.array(list(df[args.label_column])))
        
        df = pd.read_csv(args.valid_data).sample(frac=1.0, random_state=args.seed)
        X_valid = torch.tensor(np.array([tokenize(t, tokenizer, msl=args.msl) for t in tqdm(df[args.text_column])]))
        y_valid = torch.tensor(np.array(list(df[args.label_column])))

        # Dataloaders
        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        valid_set = torch.utils.data.TensorDataset(X_valid, y_valid)
        train_sampler = torch.utils.data.RandomSampler(train_set)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

        # Model
        device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
        if args.random_init:
            print("Initializing new randomly-initialized model from configuration")
            config = AutoConfig.from_pretrained(args.pretrained)
            model = AutoModelForSequenceClassification.from_config(config)
        else:
            print("Loading from pretrained checkpoint")
            model = AutoModelForSequenceClassification.from_pretrained(args.pretrained)
        _ = model.resize_token_embeddings(len(tokenizer))
        model = model.to(device)
        print("Model has {:,} trainable parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        # Initialize loss, optimizer, and scheduler
        if args.optimize_learning_rate: 
            lr = trial.suggest_loguniform('learning_rate', args.opt_lr_lowerbound, args.opt_lr_upperbound)
        else: 
            lr = args.learning_rate
        if args.optimize_weight_decay: 
            wd = trial.suggest_loguniform('weight_decay', args.opt_wd_lowerbound, args.opt_wd_upperbound)
        else: 
            wd = args.weight_decay
        
        criterion = torch.nn.CrossEntropyLoss()
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

        print("Using learning rate {} and weight decay {}".format(lr, wd), end='')
        print(" with scheduler using warmup pct {}".format(args.warmup_pct)) if args.use_scheduler else print("")

        # Train
        print("\nBegin training.")
        for e in range(1, args.epochs + 1):
            train_loss, train_acc = train(model, criterion, optimizer, train_loader, scheduler=scheduler, accumulation=args.accumulation, device=device)
            valid_loss, valid_acc = evaluate(model, criterion, valid_loader, device=device)
            
            # Pruning (hyperparam opt)
            if args.optimize_hyperparameters:
                trial.report(valid_acc, step)
                if trial.should_prune(): raise optuna.TrialPruned()

            print("Epoch {:3} | Train Loss {:.4f} | Train Acc {:.4f} | Valid Loss {:.4f} | Valid Acc {:.4f}".format(e, train_loss, train_acc, valid_loss, valid_acc))

        print("\n===== FINISHED FINETUNING =====\n")
        return valid_acc

    # Run finetuning
    # Use hyperparameter search
    if args.optimize_hyperparameters:
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
            print("    {}: {}".format(key, value))
    
    # Run standard finetuning without hyperparameter search
    else:
        _ = objective(None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--data_pct', type=float, default=1.0)
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--label_column', type=str, default='label')
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--msl', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--add_token', action='append')
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--adam_epsilon', type=float, default=1e-6)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--warmup_pct', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--optimize_hyperparameters', action='store_true')
    parser.add_argument('--opt_n_trials', type=int, default=100)
    parser.add_argument('--study_name', type=str, default='my_study')
    parser.add_argument('--optimize_seed', action='store_true')
    parser.add_argument('--opt_seed_lowerbound', type=int, default=1)
    parser.add_argument('--opt_seed_upperbound', type=int, default=99)
    parser.add_argument('--optimize_learning_rate', action='store_true')
    parser.add_argument('--opt_lr_lowerbound', type=float, default=1e-5)
    parser.add_argument('--opt_lr_upperbound', type=float, default=1e-3)
    parser.add_argument('--optimize_weight_decay', action='store_true')
    parser.add_argument('--opt_wd_lowerbound', type=float, default=1e-8)
    parser.add_argument('--opt_wd_upperbound', type=float, default=1e-3)
    args = parser.parse_args()
    print(args)

    finetune(args)

if __name__ == '__main__':
    main()