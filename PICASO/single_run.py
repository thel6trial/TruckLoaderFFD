from models.single_model import OrthrusAnt
from utils import set_logger
from datetime import datetime
import pytz
import logging
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, required=True,
                        help="max length")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="training batch size")
    parser.add_argument("--norm", type=bool, default=False,
                        help="normalize or not")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory") 
    parser.add_argument("--epoch", type=int, default=30,
                        help="training epoch")
    parser.add_argument("--training_file", type=str, required=True, default='../data/061222_train.csv')
    parser.add_argument("--valid_file", type=str, required=True, default='../data/061222_valid.csv')
    args = parser.parse_args()
    
    set_logger('./log/single_{}.log'.format(datetime.now(pytz.timezone('Asia/Singapore'))))
    logging.info(args)
    logging.info(f'Training Single Annotation Model')
    
    model = OrthrusAnt(codebert_path = 'microsoft/codebert-base', 
        decoder_layers = 6,
        fix_encoder = False, 
        beam_size = 5,
        max_source_length = args.max_length,
        max_target_length = args.max_length,
        l2_norm = args.norm,
        load_model_path = None
        # load_model_path = '/app/API-Recommendation-SO/end-to-end/output/08Des22-relevance-model-single-annotation/checkpoint-best-bleu/pytorch_model.bin'
    )
    
    # train model
    model.train(
        # train_filename ='../data/train_3_lines.csv',
        # train_filename ='../data/061222_train.csv',
        train_filename = args.training_file,
        train_batch_size = args.batch_size, 
        num_train_epochs = args.epoch, 
        learning_rate = 5e-5,
        do_eval = True, 
        # dev_filename ='../data/validate_3_lines.csv', 
        # dev_filename ='../data/061222_valid.csv', 
        dev_filename = args.valid_file,
        eval_batch_size = 32,
        output_dir = args.output_dir
    )