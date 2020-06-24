import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import numpy as np
from PIL import Image


class IAMDataset(Dataset):
    """IAM dataset."""
    def __init__(self, data_path='./iam', imsize=(64,960), min_len=1, max_len=17, num_samples=15):
        """
        Args:
            data_path (string): Path to iam dataset directory.
        """
        super(IAMDataset, self).__init__()
        self.imsize = imsize
        self.max_len = max_len
        self.min_len = min_len
        self.num_samples = num_samples
        
        self.make_data(data_path)
        
    def make_data(self, data_path):
        data = {}
        data['path'] = []
        data['transcript'] = []
        data['writer_id'] = []
        
        # Get writer ids from forms.txt
        writer_ids = {}
        with open(os.path.join(data_path, 'ascii/forms.txt'), "r") as f:
            for line in f:
                if line[0] == '#':
                    continue
                elements = line.strip().split()
                writer_ids[elements[0]] = int(elements[1])
        
        # File with image names + transcripts
        with open(os.path.join(data_path, 'ascii/words.txt'), "r") as f:
            for line in f:
                if line[0] == '#':
                    continue
                elements = line.strip().split()
                transcript = elements[-1]
                # Skip words not marked 'ok'
                if elements[1] != 'ok':
                    continue
                # Skip words that are too long or too short
                if len(transcript) > self.max_len or len(transcript) < self.min_len:
                    continue
                # Skip words with non-letter chars or upper-case
                if not transcript.isalpha() or not transcript.islower():
                    continue
                # Skip bad references / damaged files
                filename = elements[0]
                if filename in ['a01-117-05-02', 'r06-022-03-05']:
                    continue
                
                parts = filename.split('-')[:2]
                writer = "%s-%s" % tuple(parts)
                data['path'].append(os.path.join(data_path, 'words', parts[0], writer, filename + '.png'))
                data['transcript'].append(transcript)
                data['writer_id'].append(writer_ids[writer])
                
        # Force writer_ids to be 0, 1, ..., N in case of sparse indexing
        ids = np.array(data['writer_id'])
        ids_unique = np.unique(ids)
        ids = np.searchsorted(ids_unique, ids)
        data['writer_id'] = ids
        
        data['id'] = np.arange(len(data['path']))
        
        self.data = data
        self.num_writers = ids_unique.shape[0]
        
    def __len__(self):
        return len(self.data['path'])

    def __getitem__(self, i):
        # Draw additional (num_samples-1) samples with same writer_id
        same_writer_indexes = self.data['id'][self.data['writer_id'] == self.data['writer_id'][i]]
        indexes = np.random.choice(same_writer_indexes, self.num_samples-1, replace=True)
        indexes = np.append(indexes, i)
        
        images = []
        transcripts = []
        writer_id = self.data['writer_id'][i]
        for k in indexes:
            path = self.data['path'][k]
            image = Image.open(path)
            image = self.transform_image(image)

            transcript = self.data['transcript'][k]
            transcript = self.transform_text(transcript)
            
            images += [image]
            transcripts += [transcript]
            
        
        images = torch.cat(images, dim=0)
        transcripts = torch.stack(transcripts, dim=0)
        return images, transcripts, writer_id
    
    def transform_image(self, image):
        # Resize to fit the height
        size = (self.imsize[0], int(self.imsize[0] * image.size[0] / float(image.size[1])))
        image = T.functional.resize(image, size)
        # PIL to Tensor
        image = T.functional.to_tensor(image)
        # Pad to match the width
        image = nn.functional.pad(image, (0, self.imsize[1] - size[1]), value=1.)
        return image
    
    def transform_text(self, text):
        text = torch.tensor(list(map(ord, text)))
        text += 1 - ord('a')
        # Pad to match the max word len
        text = nn.functional.pad(text, (0, self.max_len - text.shape[0]))
        return text
    
    def create_word_dataset(self):
        return IAMDatasetWords(self.data['transcript'], self.max_len)
    

    
class IAMDatasetWords(Dataset):
    """Words from IAM dataset."""
    def __init__(self, words, max_len=17):
        super(IAMDatasetWords, self).__init__()
        self.words = words
        self.max_len = max_len

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        word = self.words[i]  
        word = self.transform_text(word)
        return word

    def transform_text(self, text):
        text = torch.tensor(list(map(ord, text)))
        text += 1 - ord('a')
        # Pad to match the max word len
        text = nn.functional.pad(text, (0, self.max_len - text.shape[0]))
        return text