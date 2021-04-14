from data import load_corpus
from model1a import BERTTrainer
from model1e import BERT1E
from model1f import BERT1F
from bertutil import MyBertTokenizer, PhraseAlignmentDataset, prepare_input_dual, warmup_linear
import numpy as np
from tqdm import tqdm, trange
import os, random, logging, argparse, sys
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from pytorch_pretrained_bert.optimization import BertAdam

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_type",
                    default=None,
                    type=str,
                    required=True,
                    choices=['BERTTrainer', 'BERT1E', 'BERT1F'],
                    help="Model type: BERTTrainer, BERT1E, BERT1F")
parser.add_argument("--pooling",
                    default=None,
                    type=str,
                    required=True,
                    choices=['max', 'mean'],
                    help="Pooling method: max or mean")
parser.add_argument("--lossfnc",
                    default='TripletMarginLoss',
                    type=str,
                    choices=['TripletMarginLoss'],
                    help="LossFunction: TripletMarginLoss")
parser.add_argument("--margin",
                    default=None,
                    type=float,
                    required=True,
                    help="Margin in TripletMarginLoss")
parser.add_argument("--lr",
                    default=None,
                    type=float,
                    required=True,
                    help="Learning rate")
parser.add_argument("--train_epoch",
                    default=None,
                    type=int,
                    required=True,
                    help="Max number of training epochs")
parser.add_argument("--early_stop",
                    default=None,
                    type=int,
                    required=True,
                    help="Early stopping patience")
parser.add_argument("--min_delta",
                    default=0.005,
                    type=float,
                    help="Early stopping delta")
parser.add_argument("--seed",
                    default=42,
                    type=float,
                    help="Seed for initialization")
parser.add_argument("--ft_bert", action='store_true', help="Fine-tune BERT layer")
args = parser.parse_args()


def main():
    # BERT settings
    max_seq_length = 128
    bert_model = 'bert-base-uncased'
    dropout = 0.1  # Default of run_glue.py at Transformers

    # Multi-head attention
    num_heads = 8
    res_dropout = 0.1  # Recommended in "Attention is all you need"

    # Learning settings
    warmup_proportion = 0.1
    gradient_accumulation_steps = 1
    batch_size = 16

    # set up gpu environment
    # For debugging... set gpu device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # BERT config
    do_lower_case = True if 'uncased' in bert_model else False
    bert_emb_dim = 768 if 'base' in bert_model else 1024

    ## CNN settings
    map_emb_size = bert_emb_dim

    # Model config
    config_model = {
        'bert_model': bert_model,
        'bert_emb_dim': bert_emb_dim,
        'bert_layers': [int(x) for x in "-1,-2,-3,-4".split(",")],
        'map_embed_size': map_emb_size,
        'max_seq_len': max_seq_length,
        'dropout': dropout,
        'num_heads': num_heads,
        'res_dropout': res_dropout,
        'lossfnc': args.lossfnc,
        'margin': args.margin,
        'pooling': args.pooling,
        'bsize': batch_size,
        'ft_bert': True if args.ft_bert else False,
        'device': device
    }

    # Prepare model
    model = eval(args.model_type)(config_model)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.ft_bert:
        model_name = args.model_type + '_' + args.lossfnc + '_margin-' + str(args.margin) + '_lr-' + str(
            args.lr) + '_' + args.pooling + '_' + str(
            args.train_epoch) + '_ft-' + bert_model
    else:
        model_name = args.model_type + '_' + args.lossfnc + '_margin-' + str(args.margin) + '_lr-' + str(
            args.lr) + '_' + args.pooling + '_' + str(
            args.train_epoch) + '_' + bert_model
    output_model_path = os.path.join(args.out_dir, model_name + '.pkl')

    # Bert tokenizer
    tokenizer = MyBertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

    # Load data
    s_tokens, t_tokens, s_trees, t_trees, annotator1, annotator2, annotator3 = load_corpus('../data/ESPADA/', None, True, False)
    features = prepare_input_dual(s_tokens, t_tokens, s_trees, t_trees, tokenizer, max_seq_length)

    trainloader, devloader, train, dev, train_raw_dic = separate_train_dev(
        features, s_trees, t_trees, annotator1, annotator2, annotator3, batch_size, args.lossfnc)

    t_total = int(
        len(train) / batch_size / gradient_accumulation_steps * args.train_epoch)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=warmup_proportion,
                         t_total=t_total)

    # Prepare logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Train the model
    tr_loss = 0
    global_step = 0
    nb_tr_steps = 0
    training_rec = {}
    prev_loss = sys.float_info.max
    stop_train_cnt = 0
    for epoch in trange(args.train_epoch, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_steps = 0

        # Fine-tuning of BERT
        for step, batch_idx in enumerate(tqdm(trainloader, desc="Iteration")):
            batch = train.get_batch(batch_idx)
            loss = model(batch['features'], batch['s_span_set'], batch['t_span_set'], batch['label_set'])

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.lr * warmup_linear(global_step / t_total, warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        current_loss = tr_loss / nb_tr_steps

        # Validation
        dev_loss = validation(dev, devloader, model, n_gpu, gradient_accumulation_steps)
        training_rec.update({'global_step {0:04d}'.format(epoch): global_step,
                             'train_loss {0:04d}'.format(epoch): current_loss,
                             'dev_loss {0:04d}'.format(epoch): dev_loss})
        logger.info(" train loss = %.5f", current_loss)
        logger.info(" dev loss = %.5f", dev_loss)

        if prev_loss - dev_loss <= args.min_delta:
            if dev_loss < prev_loss:
                prev_loss = dev_loss
                # Save a trained model
                torch.save(model.state_dict(), output_model_path)
            stop_train_cnt += 1
            if stop_train_cnt >= args.early_stop:
                break
        else:
            prev_loss = dev_loss
            # Save a trained model
            torch.save(model.state_dict(), output_model_path)
            stop_train_cnt = 0

        # if prev_loss - dev_loss > args.min_delta:  # dev_loss < prev_loss:
        #     # Save a trained model
        #     torch.save(model.state_dict(), output_model_path)
        #     prev_loss = dev_loss
        # else:
        #     stop_train_cnt += 1

        # if stop_train_cnt >= args.early_stop:
        #     break

        # Shuffle negative examples: Call at every epoch
        trainloader, train = shuffle_negative_sample(train_raw_dic, batch_size, args.lossfnc, True)

    with open(os.path.join(args.out_dir, model_name + '.log'), "w") as writer:
        logger.info("***** Training records *****")
        for key in sorted(training_rec.keys()):
            logger.info("  %s = %s", key, str(training_rec[key]))
            writer.write("%s = %s\n" % (key, str(training_rec[key])))


def validation(dev, devloader, model, n_gpu, gradient_accumulation_steps):
    dev_loss = 0
    nb_steps = 0
    model.eval()
    for step, batch_idx in enumerate(tqdm(devloader, desc="Iteration")):
        batch = dev.get_batch(batch_idx)
        with torch.no_grad():
            loss = model(batch['features'], batch['s_span_set'], batch['t_span_set'], batch['label_set'])

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        dev_loss += loss.item()
        nb_steps += 1

    current_loss = dev_loss / nb_steps

    return current_loss


def shuffle_negative_sample(input_raw_dic, batch_size, lossfnc, is_train):
    dataset = PhraseAlignmentDataset(input_raw_dic['features'], input_raw_dic['s_trees'], input_raw_dic['t_trees'],
                                     input_raw_dic['annotations'], lossfnc)

    if is_train:
        dataloader = DataLoader(
            dataset=TensorDataset(torch.tensor([i for i in range(len(dataset))], dtype=torch.int)),
            sampler=RandomSampler(dataset), batch_size=batch_size)
    else:
        dataloader = DataLoader(
            dataset=TensorDataset(torch.tensor([i for i in range(len(dataset))], dtype=torch.int)),
            sampler=SequentialSampler(dataset), batch_size=batch_size)

    return dataloader, dataset


def separate_train_dev(features, s_trees, t_trees, annotator1, annotator2, annotator3, batch_size, lossfnc):
    # Shuffle input
    permutation = np.random.permutation(len(features))
    s_features = [features[i] for i in permutation]
    s_s_trees = [s_trees[i] for i in permutation]
    s_t_trees = [t_trees[i] for i in permutation]
    s_annotator1 = [annotator1[i] for i in permutation]
    s_annotator2 = [annotator2[i] for i in permutation]
    s_annotator3 = [annotator3[i] for i in permutation]

    # Make 90% train set
    train_num = int(np.ceil(len(features) * 0.9))

    train_raw_dic = {
        'features': s_features[0:train_num] * 3,
        's_trees': s_s_trees[0:train_num] * 3,
        't_trees': s_t_trees[0:train_num] * 3,
        'annotations': s_annotator1[0:train_num] + s_annotator2[0:train_num] + s_annotator3[0:train_num]
    }
    trainloader, train = shuffle_negative_sample(train_raw_dic, batch_size, lossfnc, True)

    dev_raw_dic = {
        'features': s_features[train_num:] * 3,
        's_trees': s_s_trees[train_num:] * 3,
        't_trees': s_t_trees[train_num:] * 3,
        'annotations': s_annotator1[train_num:] + s_annotator2[train_num:] + s_annotator3[train_num:]
    }
    devloader, dev = shuffle_negative_sample(dev_raw_dic, batch_size, lossfnc, False)

    return trainloader, devloader, train, dev, train_raw_dic


if __name__ == "__main__":
    # execute only if run as a script
    main()
