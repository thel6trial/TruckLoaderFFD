"""
ex.: python calculate_bleu_score.py --reference codebert/saved_models_our_data/test_-1.gold --candidate codebert/saved_models_our_data/test_-1.output
python calculate_bleu_score.py --reference codebert/saved_models_our_data-10-epochs/test_-1.gold --candidate codebert/saved_models_our_data-10-epochs/test_-1.output
python calculate_bleu_score.py --reference codebert/saved_models_our_data-30-epochs/test_-1.gold --candidate codebert/saved_models_our_data-30-epochs/test_-1.output
"""

from nltk.translate.bleu_score import sentence_bleu
import argparse
import re
from icecream import ic

def argparser():
    Argparser = argparse.ArgumentParser()
    Argparser.add_argument('--reference', type=str, default='summaries.txt', help='Reference File')
    Argparser.add_argument('--candidate', type=str, default='candidates.txt', help='Candidate file')

    args = Argparser.parse_args()
    return args

args = argparser()

reference=[]
candidate=[]

if len(args.reference.split(',')) > 1:
    for f in args.reference.split(','):
        reference += open(f, 'r').readlines()
    for f in args.candidate.split(','):
        candidate += open(f, 'r').readlines()
else:
    reference = open(args.reference, 'r').readlines()
    candidate = open(args.candidate, 'r').readlines()

if len(reference) != len(candidate):
    raise ValueError('The number of sentences in both files do not match.')

score = 0.

score_1 = []
score_2 = []
score_3 = []
score_4 = []

ic(len(reference))

for i in range(len(reference)):
    bleu1 = sentence_bleu([reference[i].lower().strip().split()], candidate[i].lower().strip().split(), weights=(1,0,0,0)) 
    bleu2 = sentence_bleu([reference[i].lower().strip().split()], candidate[i].lower().strip().split(), weights=(1/2,1/2,0,0)) 
    bleu3 = sentence_bleu([reference[i].lower().strip().split()], candidate[i].lower().strip().split(), weights=(1/3,1/3,1/3,0)) 
    bleu4 = sentence_bleu([reference[i].lower().strip().split()], candidate[i].lower().strip().split(), weights=(1/4,1/4,1/4,1/4)) 

    score_1.append(bleu1)
    score_2.append(bleu2)
    score_3.append(bleu3)
    score_4.append(bleu4)


# score /= len(reference)
fin_bleu_1 = sum(score_1)/len(reference)
fin_bleu_2 = sum(score_2)/len(reference)
fin_bleu_3 = sum(score_3)/len(reference)
fin_bleu_4 = sum(score_4)/len(reference)

print(f"BLEU-1: {fin_bleu_1}")
print(f"BLEU-2: {fin_bleu_2}")
print(f"BLEU-3: {fin_bleu_3}")
print(f"BLEU-4: {fin_bleu_4}")

# print("The bleu score is: "+str(score))