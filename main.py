import os
import numpy as np
import nltk
from dataloaders.qmsum import QMSum
from pytorch_lightning import Trainer
import random
from tqdm import tqdm
from config import Config
from argparse import ArgumentParser
from utils.utils import (gpu_wrapper, rouge_with_pyrouge)
from torch.utils.data import DataLoader
from transformers import (RobertaTokenizer, RobertaForTokenClassification,
                          BartTokenizer,
                          AdamW)
from Modules.dynamic_rag import DynamicRagForGeneration
from pytorch_lightning import LightningModule
from nltk.tokenize import sent_tokenize, word_tokenize
import gc
import torch


gc.collect()
torch.cuda.empty_cache()
config = Config()
ROUND = config.ROUND
EPSILON = 1e-10
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if config.gpu:
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


retriever_tokenizer = RobertaTokenizer.from_pretrained(config.retriever_name_or_path)
generator_tokenizer = BartTokenizer.from_pretrained(config.generator_name_or_path)

train_set = QMSum('train', retriever_tokenizer=retriever_tokenizer, generator_tokenizer=generator_tokenizer)
val_set = QMSum('valid', retriever_tokenizer=retriever_tokenizer, generator_tokenizer=generator_tokenizer)
test_set = QMSum('test', retriever_tokenizer=retriever_tokenizer, generator_tokenizer=generator_tokenizer)

train_dataloader = DataLoader(train_set,
                                    batch_size=config.train_batch_size // config.gradient_accumulation_steps,
                                    shuffle=True,
                                    num_workers=config.num_workers)
    
eval_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=config.num_workers)
        
test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config.num_workers)


class MyLightningModule(LightningModule):
    def __init__(self, retriever_tokenizer, generator_tokenizer, train_set, val_set, test_set):
        super().__init__()
        self.automatic_optimization = False
        self.retriever_tokenizer=retriever_tokenizer
        self.generator_tokenizer=generator_tokenizer

        # Load retriever model.
        self.retriever = RobertaForTokenClassification.from_pretrained(config.retriever_name_or_path,
                                                                    num_labels=1
                                                                    ,gradient_checkpointing=True
                                                                    )
        # Load generator model.
        self.generator = DynamicRagForGeneration.from_pretrained(config.generator_name_or_path,
                                                                n_docs=config.top_k
                                                                ,gradient_checkpointing=True
                                                                )
        # Load loss.
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='none')

        self.modules = ['retriever', 'generator', 'criterion_cls']
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        for module in self.modules:
            print('--- {}: '.format(module))
            print(getattr(self, module))
            if getattr(self, module) is not None:
                setattr(self, module, gpu_wrapper(getattr(self, module)))

        self.scopes = {'cls': ['retriever'], 'gen': ['generator']}
        for scope in self.scopes.keys():
            setattr(self, scope + '_lr', getattr(config, scope + '_lr'))

        self.iter_num = 0
        self.best_metric = - float('inf')
        self.decay_num = 0
        self.no_improvement = 0

        # Tokenization for BLEU.
        nltk_wordpunk_tokenizer = nltk.tokenize.WordPunctTokenizer()
        self.bleu_tokenizer = lambda x: nltk_wordpunk_tokenizer.tokenize(x)

    def configure_optimizers(self):
        for scope in self.scopes.keys():
            optimizer_grouped_parameters = [
                {'params': [],
                'weight_decay': config.weight_decay},
                {'params': [],
                'weight_decay': 0.0},
                
            ]
            no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

            for module in self.scopes[scope]:
                if getattr(self, module) is not None:
                    for n, p in getattr(self, module).named_parameters():
                        # k is the parameter name; v is the parameter value.
                        if p.requires_grad:
                            # Weight decay.
                            if not any(nd in n for nd in no_decay):
                                print("[{} Trainable:]".format(module), n)
                                optimizer_grouped_parameters[0]['params'].append(p)
                            else:
                                print("[{} Trainable (bias/LN):]".format(module), n)
                                optimizer_grouped_parameters[1]['params'].append(p)
                        else:
                            print("[{} Frozen:]".format(module), n)

            if config.optimizer == 'adam':
                setattr(self, scope + '_optim', AdamW(optimizer_grouped_parameters, lr=getattr(self, scope + '_lr')))
            else:
                raise ValueError()

            setattr(self,
                    scope + '_grouped_parameters',
                    optimizer_grouped_parameters[0]['params'] + optimizer_grouped_parameters[1]['params'])
        return self.cls_optim, self.gen_optim

    def forward(self, data):
        retriever_input_ids, global_attention_mask, cls_ids, oracle, \
            context_input_ids, context_attention_mask, labels, index = data

        assert retriever_input_ids.shape[0] == 1

        num_oracle = len(oracle[0])
        
        retriever_outputs = self.retriever(input_ids=retriever_input_ids.squeeze(0),
                                            output_hidden_states=True)
        retriever_all_logits = retriever_outputs.logits
        retriever_all_logits = retriever_all_logits.squeeze(2)
        retriever_cls_logits = retriever_all_logits.contiguous().view(-1)[cls_ids.squeeze(0).cpu().tolist()].unsqueeze(0)
        
        # Retrieval loss.
        ret_loss = 0
        for turn_id in oracle[0].cpu().tolist():
            ret_loss = ret_loss + self.criterion_cls(input=retriever_cls_logits,
                                                        target=gpu_wrapper(torch.LongTensor([turn_id])))

        if num_oracle > 0:
            ret_loss = ret_loss / num_oracle

        # Generation loss.
        if config.loss_alpha != 0 or num_oracle == 0:
            if oracle.shape[1] != 0:
                k = min(config.top_k, retriever_cls_logits.shape[1])
                retriever_topk_indices = oracle.squeeze(0).cpu().tolist()[:k]
                if config.hybrid_train and len(retriever_topk_indices) < k:
                    _, real_retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                    real_retriever_topk_indices = real_retriever_topk_indices[0].cpu().tolist()
                    # assert 1==0, print(real_retriever_topk_indices)
                    retriever_topk_indices = retriever_topk_indices + [idx for idx in real_retriever_topk_indices if idx not in retriever_topk_indices]
                    retriever_topk_indices = retriever_topk_indices[:k]
                doc_scores = retriever_cls_logits[:, retriever_topk_indices]
            else:
                doc_scores, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                retriever_topk_indices = retriever_topk_indices[0].cpu().tolist()

            if len(retriever_topk_indices) < config.top_k:
                doc_scores = torch.cat([doc_scores, gpu_wrapper(torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (config.top_k - len(retriever_topk_indices))

            generator_outputs = self.generator(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                doc_scores=doc_scores,
                                                labels=labels)  
                
            return generator_outputs, ret_loss

    def training_step(self, batch, batch_idx):
        generator_outputs, ret_loss = self.forward(batch)
        
        seq_loss = generator_outputs.loss
        consistency_loss = generator_outputs.consistency_loss

        tot_loss = seq_loss * config.loss_alpha + ret_loss
        tot_loss = tot_loss + config.consistency_alpha * consistency_loss

        # Backward.
        if config.gradient_accumulation_steps > 1:
            tot_loss = tot_loss / config.gradient_accumulation_steps
        logs = {'train_loss': tot_loss}
        return {'loss': tot_loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        if self.iter_num % (config.save_steps * config.gradient_accumulation_steps) == 0:
            if config.target_task in ['govreport-latent', 'arxiv-latent']:
                    beam_size = 1  # Use beam_size = 1 for validation
            else:
                    beam_size = 5
            
        predictions = []
        topks = []
        doc_scoreses = []
        the_set = self.val_set

        retriever_input_ids, global_attention_mask, cls_ids, oracle, \
            context_input_ids, context_attention_mask, labels, index = batch

        assert retriever_input_ids.shape[0] == 1

        # Forward (prediction).
        with torch.no_grad():
            retriever_outputs = self.retriever(input_ids=retriever_input_ids.squeeze(0),
                                            output_hidden_states=True)
            retriever_all_logits = retriever_outputs.logits
            retriever_all_logits = retriever_all_logits.squeeze(2)
            retriever_cls_logits = retriever_all_logits.contiguous().view(-1)[cls_ids.squeeze(0).cpu().tolist()].unsqueeze(0)  # shape = (1, n_turns)

            if config.oracle_test and oracle.shape[1] != 0:
                k = min(config.top_k, retriever_cls_logits.shape[1])
                doc_scores = torch.gather(retriever_cls_logits, dim=1, index=oracle)[:, :k]
                retriever_topk_indices = oracle.squeeze(0).cpu().tolist()[:k]
            else:
                doc_scores, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                retriever_topk_indices = retriever_topk_indices[0].cpu().tolist()

            if len(retriever_topk_indices) < config.top_k:
                doc_scores = torch.cat([doc_scores, gpu_wrapper(torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (config.top_k - len(retriever_topk_indices))

            if config.loss_alpha != 0:
                outputs = self.generator.generate(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                doc_scores=doc_scores,
                                                num_beams=beam_size,
                                                min_length=config.min_length,
                                                max_length=config.max_target_len,
                                                no_repeat_ngram_size=config.no_repeat_ngram_size,
                                                length_penalty=config.length_penalty,
                                                )
                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape[0] == 1

                # Predictions:
                decoded_pred = self.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                cleaned_prediction = ["\n".join(sent_tokenize(" ".join(word_tokenize(pred)))) for pred in decoded_pred]
            else:
                cleaned_prediction = ["empty prediction because loss_alpha = 0."]
            predictions.extend(cleaned_prediction)

            # top_k:
            decoded_topk = self.generator_tokenizer.batch_decode(context_input_ids[:, retriever_topk_indices].contiguous().view(config.top_k, -1), skip_special_tokens=True)
            if config.target_task in ['govreport-latent', "arxiv-latent"]:
                cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                topks.append(cleaned_topk)

            else:
                cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent[sent.index(':') + 1:sent.index(' // ') if ' // ' in sent else len(sent)] for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                topks.append(cleaned_topk)

            doc_scoreses.append(doc_scores[0])

        # Load references.
        references = ["\n".join(sent_tokenize(" ".join(word_tokenize(sent)))) for sent in the_set.get_references()]

        # ROUGE.
        rouge1, rouge2, rougeL = rouge_with_pyrouge(preds=predictions, refs=references)
        print(rouge1, rouge2, rougeL)

        rouge1_topk, rouge2_topk, rougeL_topk = rouge_with_pyrouge(preds=topks, refs=references)

        print(rouge1_topk, rouge2_topk, rougeL_topk)

        if config.loss_alpha != 0:
            metric = rouge1 + rouge2 + rougeL
        else:
            metric = rouge1_topk + rouge2_topk + rougeL_topk

        if not test and metric > self.best_metric:
            self.best_metric = metric
            self.save_step(['retriever', 'generator'])

            peep_num = 3
            for sent_id in range(peep_num):
                print('Pred:\n{}'.format(predictions[sent_id]))
                print('-' * 20)
                print('topk:\n{}'.format(topks[sent_id]))
                print('-' * 20)
                print('Ref:\n{}'.format(references[sent_id]))
                print('-' * 20)
                print()
                print('=' * 50)

        self.set_training(mode=True)

        base_name = '{}.gen'.format('test' if test else 'valid')
        save_path = os.path.join(config.sample_dir, base_name)
        torch.save((predictions, references), save_path)
        return {"Rouge-1": rouge1, "Rouge-2": rouge2, "Rouge-L": rougeL}
        score = self.seq_evaluate_gen(test=False, beam_size=beam_size, batch=batch)
                        # Learning rate decay.
        return score

    def test_step(self, batch, batch_idx):
        # Evaluate.
        if config.target_task in ['qmsum-latent',
                                'arxiv-latent',
                                'govreport-latent',
                                ]:
            if config.target_task in ['govreport-latent']:
                beam_size = 4  # Use beam_size = 1 for validation
            elif config.target_task in ['arxiv-latent']:
                beam_size = 5
            elif config.target_task in ['qmsum']:
                beam_size = 6
            else:
                beam_size = 1

        predictions = []
        topks = []
        doc_scoreses = []
        the_set = self.test_set

        retriever_input_ids, global_attention_mask, cls_ids, oracle, \
            context_input_ids, context_attention_mask, labels, index = batch

        assert retriever_input_ids.shape[0] == 1

        # Forward (prediction).
        with torch.no_grad():
            retriever_outputs = self.retriever(input_ids=retriever_input_ids.squeeze(0),
                                            output_hidden_states=True)
            retriever_all_logits = retriever_outputs.logits
            retriever_all_logits = retriever_all_logits.squeeze(2)
            retriever_cls_logits = retriever_all_logits.contiguous().view(-1)[cls_ids.squeeze(0).cpu().tolist()].unsqueeze(0)  # shape = (1, n_turns)

            if config.oracle_test and oracle.shape[1] != 0:
                k = min(config.top_k, retriever_cls_logits.shape[1])
                doc_scores = torch.gather(retriever_cls_logits, dim=1, index=oracle)[:, :k]
                retriever_topk_indices = oracle.squeeze(0).cpu().tolist()[:k]
            else:
                doc_scores, retriever_topk_indices = torch.topk(retriever_cls_logits, k=min(config.top_k, retriever_cls_logits.shape[1]), dim=1)
                retriever_topk_indices = retriever_topk_indices[0].cpu().tolist()

            if len(retriever_topk_indices) < config.top_k:
                doc_scores = torch.cat([doc_scores, gpu_wrapper(torch.zeros((1, config.top_k - len(retriever_topk_indices)))).fill_(-float('inf'))], dim=1)
                retriever_topk_indices = retriever_topk_indices + [retriever_topk_indices[-1]] * (config.top_k - len(retriever_topk_indices))

            if config.loss_alpha != 0:
                outputs = self.generator.generate(context_input_ids=context_input_ids[:, retriever_topk_indices].contiguous().view(context_input_ids.shape[0] * config.top_k, -1),
                                                context_attention_mask=context_attention_mask[:, retriever_topk_indices].contiguous().view(context_attention_mask.shape[0] * config.top_k, -1),
                                                doc_scores=doc_scores,
                                                num_beams=beam_size,
                                                min_length=config.min_length,
                                                max_length=config.max_target_len,
                                                no_repeat_ngram_size=config.no_repeat_ngram_size,
                                                length_penalty=config.length_penalty,
                                                )
                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape[0] == 1

                # Predictions:
                decoded_pred = self.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

                cleaned_prediction = ["\n".join(sent_tokenize(" ".join(word_tokenize(pred)))) for pred in decoded_pred]
            else:
                cleaned_prediction = ["empty prediction because loss_alpha = 0."]
            predictions.extend(cleaned_prediction)

            # top_k:
            decoded_topk = self.generator_tokenizer.batch_decode(context_input_ids[:, retriever_topk_indices].contiguous().view(config.top_k, -1), skip_special_tokens=True)
            if config.target_task in ['govreport-latent', "arxiv-latent"]:
                cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                topks.append(cleaned_topk)

            else:
                cleaned_topk = "\n".join(sent_tokenize(" ".join(word_tokenize(" ".join([sent[sent.index(':') + 1:sent.index(' // ') if ' // ' in sent else len(sent)] for sent, prob in zip(decoded_topk, torch.softmax(doc_scores[0], dim=0)) if prob > 1e-10])))))
                topks.append(cleaned_topk)

            doc_scoreses.append(doc_scores[0])

        # Load references.
        references = ["\n".join(sent_tokenize(" ".join(word_tokenize(sent)))) for sent in the_set.get_references()]

        # ROUGE.
        rouge1, rouge2, rougeL = rouge_with_pyrouge(preds=predictions, refs=references)
        print(rouge1, rouge2, rougeL)

        rouge1_topk, rouge2_topk, rougeL_topk = rouge_with_pyrouge(preds=topks, refs=references)

        print(rouge1_topk, rouge2_topk, rougeL_topk)

        if config.loss_alpha != 0:
            metric = rouge1 + rouge2 + rougeL
        else:
            metric = rouge1_topk + rouge2_topk + rougeL_topk

        if not test and metric > self.best_metric:
            self.best_metric = metric
            self.save_step(['retriever', 'generator'])

            peep_num = 3
            for sent_id in range(peep_num):
                print('Pred:\n{}'.format(predictions[sent_id]))
                print('-' * 20)
                print('topk:\n{}'.format(topks[sent_id]))
                print('-' * 20)
                print('Ref:\n{}'.format(references[sent_id]))
                print('-' * 20)
                print()
                print('=' * 50)

        self.set_training(mode=True)

        base_name = '{}.gen'.format('test' if test else 'valid')
        save_path = os.path.join(config.sample_dir, base_name)
        torch.save((predictions, references), save_path)
        return {"Rouge-1": rouge1, "Rouge-2": rouge2, "Rouge-L": rougeL}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--distributed_backend", default=None)
    args = parser.parse_args()

    model = MyLightningModule(retriever_tokenizer, generator_tokenizer, train_set, val_set, test_set)
    trainer = Trainer(accelerator=args.accelerator, gpus=args.gpus, distributed_backend=args.distributed_backend, num_sanity_val_steps=0)
    trainer.fit(model, train_dataloader, eval_dataloader)
