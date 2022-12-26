import os
import sys
sys.path.insert(1, '/tmp/pycharm_project/DYLE/utils')
sys.path.insert(1, '/tmp/pycharm_project/DYLE/')
sys.path.insert(1, '/tmp/pycharm_project/DYLE/dataloaders')
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from unified_data import DialSumBase
from config import Config

config = Config()

MAP = {'train': 'train', 'test': 'test'}


class LG(DialSumBase):
    """The QMSum dataset."""

    def __init__(self, mode, retriever_tokenizer, generator_tokenizer):
        super(LG, self).__init__(mode, retriever_tokenizer, generator_tokenizer)
        self.root = os.path.join('dataset', 'QMSum')

        # TODO: add qmsum_ before filename, see arxiv.py
        self.cached_features_file = os.path.join(self.root, '{}_cached_qmsum'.format(MAP[self.mode]))

        self.file_name = os.path.join(self.root, '{}.jsonl'.format(MAP[self.mode]))

        self.load_features_from_cache()

    def get_features(self):
        self.features = self.read_dialogue_summarization1()
        print('QMSum data successfully read.')

