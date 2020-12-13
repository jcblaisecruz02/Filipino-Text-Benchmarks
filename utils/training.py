import torch
import torch.utils.data as data

from .data import process_data

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import hashlib

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

try:
    from apex import amp 
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def accuracy(out, y): 
    # Columnwise mean accurarcy for multilabel classification
    if len(y.shape) > 1:
        return ((out.sigmoid() > 0.5).long() == y).float().mean(0).mean()
    # Standard accuracy
    return (out.argmax(1) == y).sum().item() / len(y)

# Train one epoch
def train(model, criterion, optimizer, train_loader, scheduler=None, accumulation=1, device=None, fp16=False):
    model.train()
    train_loss, train_acc = 0, 0
    for i, batch in enumerate(tqdm(train_loader)):
        
        if 'distilbert' in str(type(model)) or 'roberta' in str(type(model)):
            x, attention_mask, y = batch
            x, y = x.to(device), y.to(device)
            attention_mask = attention_mask.to(device)
            out = model(input_ids=x, attention_mask=attention_mask)[0]
        else:
            x, token_type_ids, attention_mask, y = batch
            x, y = x.to(device), y.to(device)
            token_type_ids, attention_mask = token_type_ids.to(device), attention_mask.to(device)
            out = model(input_ids=x, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        loss = criterion(out, y)

        if fp16 and APEX_AVAILABLE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
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

# Evaluate one epoch
def evaluate(model, criterion, valid_loader, device=None):
    model.eval()
    valid_loss, valid_acc = 0, 0
    for batch in tqdm(valid_loader):
        if 'distilbert' in str(type(model)) or 'roberta' in str(type(model)):
            x, attention_mask, y = batch
            x, y = x.to(device), y.to(device)
            attention_mask = attention_mask.to(device)
            with torch.no_grad():
                out = model(input_ids=x, attention_mask=attention_mask)[0]
        else:
            x, token_type_ids, attention_mask, y = batch
            x, y = x.to(device), y.to(device)
            token_type_ids, attention_mask = token_type_ids.to(device), attention_mask.to(device)
            with torch.no_grad():
                out = model(input_ids=x, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        
        with torch.no_grad():
            loss = criterion(out, y)

        valid_loss += loss.item()
        valid_acc += accuracy(out, y)
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)

    return valid_loss, valid_acc

# Run full finetuning
def run_finetuning(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Get text columns
    t_columns = args.text_columns.split(',')
    num_texts = len(t_columns)
    if num_texts == 1: t_columns = t_columns[0]

    # Get label columns
    l_columns = args.label_columns.split(',')
    num_labels = len(l_columns)
    if num_labels == 1: l_columns = l_columns[0]

    if args.fp16 and not APEX_AVAILABLE:
        print("FP16 toggle is on but Apex is not available. Using FP32 training.")

    if args.do_train:
        # Configure tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
        if args.add_token != '':
            add_token = {'additional_special_tokens': args.add_token.split(',')}
            added = tokenizer.add_special_tokens(add_token)
            
        print('\n' + '=' * 50, '\nCONFIGURE FINETUNING SETUP', '\n' + '=' * 50)
        if args.add_token != '': print("Addded {} special tokens:".format(added), args.add_token)

        # Produce hash code for cache
        f_string = args.train_data + args.valid_data + str(args.msl) + str(args.seed) + args.pretrained + str(args.data_pct)
        hashed = 'cache_' + hashlib.md5(f_string.encode()).hexdigest() + '.pt'

        # Produce the dataset if cache doesn't exist
        if hashed not in os.listdir() or args.retokenize_data:
            print("Producing dataset cache. This will take a while.")
            s = time.time()

            df = pd.read_csv(args.train_data, lineterminator='\n').sample(frac=args.data_pct, random_state=args.seed)
            text, labels = df[t_columns].values, df[l_columns].values
            train_dataset = process_data(text, labels, tokenizer, msl=args.msl)

            df = pd.read_csv(args.valid_data, lineterminator='\n')
            text, labels = df[t_columns].values, df[l_columns].values
            valid_dataset = process_data(text, labels, tokenizer, msl=args.msl)

            if args.save_cache:
                print('Saving data cache')
                with open(hashed, 'wb') as f:
                    torch.save([train_dataset, valid_dataset], f)

            print("Preprocessing finished. Time elapsed: {:.2f}s".format(time.time() - s))

        # Load the dataset if the cache exists
        else:
            print('Cache found. Loading training and validation data.')
            with open(hashed, 'rb') as f:
                train_dataset, valid_dataset = torch.load(f)

        # Produce dataloaders
        train_sampler = data.RandomSampler(train_dataset)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
        valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        # Configure model
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

        # Configure loss function
        criterion = torch.nn.CrossEntropyLoss() if num_labels == 1 else torch.nn.BCEWithLogitsLoss()

        # Configure optimizer
        if args.optimizer == 'adam':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
                                            "weight_decay": args.weight_decay}, 
                                            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
                                            "weight_decay": 0.0}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            optimizer.zero_grad()
        elif args.optimizer == 'lamb':
            from pytorch_lamb import Lamb
            optimizer = Lamb(model.parameters(), 
                             lr=args.learning_rate, 
                             weight_decay=args.weight_decay,
                             betas=(args.adam_b1, args.adam_b2))

        # Configure FP16
        if args.fp16 and APEX_AVAILABLE:
            print("Using FP16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

        # Configure scheduler
        if args.use_scheduler:
            steps = len(train_loader) * args.epochs // args.accumulation
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps * args.warmup_pct), num_training_steps=steps)
        else: scheduler = None

        print("Using learning rate {:.4E} and weight decay {:.4E}".format(args.learning_rate, args.weight_decay), end='')
        print(" with scheduler using warmup pct {}".format(args.warmup_pct)) if args.use_scheduler else print("")

        # Training proper
        print('\n' + '=' * 50, '\nTRAINING', '\n' + '=' * 50)
        print("Training batches: {} | Validation batches: {}".format(len(train_loader), len(valid_loader)))
        for e in range(1, args.epochs + 1):
            train_loss, train_acc = train(model, criterion, optimizer, train_loader, scheduler=scheduler, accumulation=args.accumulation, device=device, fp16=args.fp16)
            valid_loss, valid_acc = evaluate(model, criterion, valid_loader, device=device)
            print("Epoch {:3} | Train Loss {:.4f} | Train Acc {:.4f} | Valid Loss {:.4f} | Valid Acc {:.4f}".format(e, train_loss, train_acc, valid_loss, valid_acc))

            # Save the model
            model.save_pretrained(args.checkpoint)
            tokenizer.save_pretrained(args.checkpoint)
            #with open(args.checkpoint, 'wb') as f:
            #    torch.save(model.state_dict(), f)

    if args.do_eval:
        print('\n' + '=' * 50, '\nBEGIN EVALUATION PROPER', '\n' + '=' * 50)

        # Load saved tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

        # Produce hash code for test cache
        f_string = args.test_data + str(args.msl) + str(args.seed) + args.pretrained
        hashed = 'cache_' + hashlib.md5(f_string.encode()).hexdigest() + '.pt'

        # Produce the dataset if cache doesn't exist
        if hashed not in os.listdir() or args.retokenize_data:
            print("Producing test data cache. This will take a while.")
            s = time.time()

            df = pd.read_csv(args.test_data, lineterminator='\n')
            text, labels = df[t_columns].values, df[l_columns].values
            test_dataset = process_data(text, labels, tokenizer, msl=args.msl)

            if args.save_cache:
                print('Saving data cache')
                with open(hashed, 'wb') as f:
                    torch.save(test_dataset, f)

            print("Preprocessing finished. Time elapsed: {:.2f}s".format(time.time() - s))

        # Load the dataset if the cache exists
        else:
            print('Cache found. Loading test data.')
            with open(hashed, 'rb') as f:
                test_dataset = torch.load(f)

        # Dataloaders
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Produce the model
        print("Loading finetuned checkpoint")
        model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss() if num_labels == 1 else torch.nn.BCEWithLogitsLoss()

        # Testing proper
        print('\n' + '=' * 50, '\nTESTING', '\n' + '=' * 50)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device=device)
        print("Test Loss {:.4f} | Test Accuracy {:.4f}".format(test_loss, test_acc))

    # Logging
    if not args.do_train: train_loss, train_acc, valid_loss, valid_acc = None, None, None, None
    if not args.do_eval: test_loss, test_acc = None, None
    return train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc
