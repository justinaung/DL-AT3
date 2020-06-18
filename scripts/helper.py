from glob import glob
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from string import punctuation
import pickle

from tensorflow import keras
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm


def load_doc(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read()


def load_descriptions(doc: str) -> Dict[str, List[str]]:
    """
    Create a mapping of image ids with descriptions from
    the Flickr8k.token.txt
    """
    mapping = defaultdict(list)

    for i, line in enumerate(doc.split('\n')):
        tokens = line.split('\t')
        if len(tokens) != 2:
            continue

        image_file_name, image_desc = tokens[0], tokens[1]
        image_id = image_file_name.split('.')[0]

        mapping[image_id].append(image_desc)

    return dict(mapping)


def clean_descriptions(descriptions: Dict[str, List[str]]):
    """
    Remove punctuations, lower case, remove hanging a and s,
    keep alpahbest only
    """
    table = str.maketrans('', '', punctuation)
    for k, desc_list in descriptions.items():
        for i, desc in enumerate(desc_list):
            # tokenize
            desc = desc.split()
            # lower case and remove punctuation
            desc = [word.lower().translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)


def save_descriptions(descriptions: Dict[str, List[str]], file_name: str):
    """
    Save lines of image_id spaced with each caption to a file
    """
    lines = [
        ''.join([image_id, ' ', desc])
        for image_id, desc_list in descriptions.items()
        for desc in desc_list
    ]
    data = '\n'.join(lines)
    with open(file_name, 'w') as f:
        f.write(data)


def to_vocabulary(descriptions: Dict[str, List[str]]) -> Set[str]:
    """
    Create a set of word vocabularies from the image descriptions
    """
    all_desc = set()
    for image_id, desc_list in descriptions.items():
        for desc in desc_list:
            all_desc.update(desc.split())
    return all_desc


def load_image_ids(filename: str) -> Set[str]:
    doc = load_doc(filename)
    image_ids = {
        line.split('.')[0]
        for line in doc.split('\n')
        if len(line) >= 1
    }
    return image_ids


def load_train_dev_test_image_paths(images_path_pattern: str, caption_dataset_path: str) -> Tuple[List[str]]:
    image_path_list = glob(images_path_pattern)

    train_images_path_list = get_common_image_path_list(
        image_path_list,
        f'{caption_dataset_path}/Flickr_8k.trainImages.txt',
    )
    dev_images_path_list = get_common_image_path_list(
        image_path_list,
        f'{caption_dataset_path}/Flickr_8k.devImages.txt',
    )
    test_images_path_list = get_common_image_path_list(
        image_path_list,
        f'{caption_dataset_path}/Flickr_8k.testImages.txt',
    )
    return train_images_path_list, dev_images_path_list, test_images_path_list


def get_common_image_path_list(image_path_list: List[str], splitted_image_names: List[str]) -> List[str]:
    with open(splitted_image_names) as f:
        splitted_image_ids = f.read().strip().split('\n')

    return [
        image_path for image_path in image_path_list
        if image_path.split('/')[-1] in splitted_image_ids
    ]


def preprocess_image(image_path, target_size=(299, 299)):
    # Convert image to appropriate size expected by a CNN model
    image = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    # Convert PIL image to numpy array of 3-dimensions
    img_arr = keras.preprocessing.image.img_to_array(image)
    # Add one more dimension
    img_arr = np.expand_dims(img_arr, axis=0)
    # Preprocess the images using method from inception V3
    img_arr = preprocess_input(img_arr) # standardize the values between -1 and 1
    return img_arr


def encode_image(image_path, cnn_model, target_size):
    image_array = preprocess_image(image_path, target_size)
    feature_vectors = cnn_model.predict(image_array)
    # reshape from (w, h) to (h, )
    feature_vectors = np.reshape(feature_vectors, feature_vectors.shape[1])
    return feature_vectors


def extract_features_from_images(image_path_list: List[str], cnn_model, target_size, save_file_path: str):
    image_encodings = dict()
    for path in tqdm(image_path_list, desc='Extracting features from images'):
        image_id = path.split('/')[-1].split('.')[0]
        image_encodings[image_id] = encode_image(path, cnn_model, target_size)

    with open(save_file_path, 'wb') as f:
        pickle.dump(image_encodings, f)
        print(f'Image encodings saved at: {save_file_path}')

    return image_encodings


def add_start_end_tokens(full_descriptions_filename: str, splitted_image_ids: List[str]) -> Dict[str, List[str]]:
    doc = load_doc(full_descriptions_filename)
    descriptions = defaultdict(list)

    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], ' '.join(tokens[1:])
        if image_id in splitted_image_ids:
            desc = ' '.join(['<START>', image_desc, '<END>'])
            descriptions[image_id].append(desc)

    return dict(descriptions)


def data_generator(
    descriptions: Dict[str, List], image_features: Dict[str, np.array], wordtoix: Dict[str, int],
    max_length: int, vocab_size: int, num_images_per_epoch: int
):
    """
    The data generator will yield:
        - X1: the extract features of an image (image_feature),
        - X2: the accumulated word sequences of a description related to the image (in_seq)
        -  y: the next encoded word sequence (out_seq)
    X2 is padded with zeros up to max_length in order to keep the same dimension for all X2s.
    """
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for image_id, desc_list in descriptions.items():
            n += 1
            image_feature = image_features[image_id]
            for desc in desc_list:
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(image_feature)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n == num_images_per_epoch:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n = 0
