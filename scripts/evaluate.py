from collections import Counter
import argparse

from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

from helper import *


FLICKR8K_TEXT = 'Flickr8k_text'
FLICKR8K_DATASET = 'Flicker8k_Dataset'


parser = argparse.ArgumentParser()
model_file_path = parser.add_argument('model_file_path', metavar='path', type=str)
beam_width = parser.add_argument('beam_width', type=int)
args = parser.parse_args()

train_img_paths, dev_img_paths, test_img_paths = load_train_dev_test_image_paths(
    f'{FLICKR8K_DATASET}/*.jpg',
    FLICKR8K_TEXT,
)

train_image_ids = [img_path.split('/')[-1].split('.')[0] for img_path in train_img_paths]
train_descriptions = add_start_end_token('notebooks/descriptions.txt', train_image_ids)

train_image_features = pickle.load(open("notebooks/train_image_encodings.pickle", "rb"))
all_train_captions = [desc
                    for desc_list in train_descriptions.values()
                    for desc in desc_list]

tokenizer = Tokenizer(oov_token='<unk>',
                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(all_train_captions)
max_length = max(len(t) for t in tokenizer.texts_to_sequences(all_train_captions))
vocab_size = len(tokenizer.index_word) + 1


def get_one_with_highest_log_prob(captions, log_prob):
    return sorted([(c, p) for c, p in zip(captions, log_prob)], key=lambda x: x[1])[-1]


def bleu_evalulate(model, descriptions, image_features, beam_width=3):

    actual, predicted = list(), list()
    # step over the whole dataset
    for image_id, desc_list in tqdm(descriptions.items(), desc=f'Evaluating model with BLEU using Beam Search with {beam_width} beam_width'):
        captions, log_probs = generate_caption_beam(
            model,
            tokenizer,
            image_features[image_id].reshape((1, 2048)),
            max_length,
            vocab_size,
            beam_width,
        )
        pred_cap, prob = get_one_with_highest_log_prob(captions, log_probs)

        # store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(pred_cap.split())

    # calculate BLEU scores
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


test_image_ids = [img_path.split('/')[-1].split('.')[0] for img_path in test_img_paths]
test_descriptions = load_splitted_descriptions('notebooks/descriptions.txt', test_image_ids)
test_image_features = pickle.load(open('notebooks/test_image_encodings.pickle', "rb"))

model = load_model(args.model_file_path)
bleu_evalulate(model, test_descriptions, test_image_features, args.beam_width)
