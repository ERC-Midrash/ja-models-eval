"""
Extracted SWEET (Subword Edit Error Tagger) inference functions.

Provides post-correction for Arabic text using a fine-tuned BertForTokenClassification
model from HuggingFace (CAMeL-Lab/text-editing-qalb14-nopnx).

The SubwordEdit class and rewrite/detokenize functions are extracted from the
CAMeL-Lab/text-editing repository (MIT License, Copyright 2025 New York University
Abu Dhabi). See: https://github.com/CAMeL-Lab/text-editing

Usage:
    from arabic.sweet_rewrite import load_sweet_model, predict
    model, tokenizer = load_sweet_model()
    corrected = predict(model, tokenizer, "يجب الإهتمام ب الصحه".split())
"""

import re
import json

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForTokenClassification


DEFAULT_MODEL_NAME = "CAMeL-Lab/text-editing-qalb14-nopnx"
# DEFAULT_MODEL_NAME = "CAMeL-Lab/text-editing-coda"


# ---------------------------------------------------------------------------
# SubwordEdit class (from edits/edit.py in CAMeL-Lab/text-editing, MIT License)
# ---------------------------------------------------------------------------

class SubwordEdit:
    def __init__(self, subword, raw_subword, edit):
        self.subword = subword
        self.raw_subword = raw_subword
        self.edit = edit

    def apply(self, subword):
        if self.edit == 'K':
            return subword

        if self.edit.startswith('KA'):
            return self._apply_append(subword, keep=True)

        if self.edit.startswith('DA'):
            return self._apply_append(subword, keep=False)

        _subword = subword.replace('##', '')
        char_edits = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|D\*|.', self.edit)
        edited_subword = self._apply_char_edits(_subword, char_edits)

        return '##' + edited_subword if '##' in subword else edited_subword

    def _apply_append(self, subword, keep=True):
        ops = re.findall(r'A_\[.*?\]+', self.edit)
        inserts = [re.sub(r'A_\[(.*?)\]', r'\1', op) for op in ops]
        return subword + ' ' + ' '.join(inserts) if keep else ''.join(inserts)

    def _apply_char_edits(self, subword, char_edits):
        edited_subword = ''
        idx = 0

        for i, char_edit in enumerate(char_edits):
            if char_edit == 'K':
                edited_subword += subword[idx]
                idx += 1

            elif char_edit == 'D':
                idx += 1

            elif char_edit.startswith('I'):
                edited_subword += re.sub(r'I_\[(.*?)\]', r'\1', char_edit)

            elif char_edit.startswith('A'):
                edited_subword += (' ' + re.sub(r'A_\[(.*?)\]', r'\1', char_edit) if i
                                   else re.sub(r'A_\[(.*?)\]', r'\1', char_edit) + ' ')

            elif char_edit == 'K*':
                chars_to_keep = self._apply_keep_star(''.join(subword[idx:]), char_edits, i + 1)
                idx += len(chars_to_keep)
                edited_subword += chars_to_keep

            elif char_edit == 'D*':
                idx += self._apply_delete_star(''.join(subword[idx:]), char_edits, i + 1)

            elif char_edit.startswith('R'):
                edited_subword += re.sub(r'R_\[(.*?)\]', r'\1', char_edit)
                idx += 1

        return edited_subword

    def _apply_keep_star(self, subword, char_edits, edit_idx):
        remaining_edits = char_edits[edit_idx:]
        inserts = [x for x in remaining_edits if (x.startswith('I') or x.startswith('A'))]

        if len(inserts) == len(remaining_edits):
            return ''.join(subword[:])
        else:
            return ''.join(subword[: -(len(remaining_edits) - len(inserts))])

    def _apply_delete_star(self, subword, char_edits, edit_idx):
        remaining_edits = char_edits[edit_idx:]
        inserts_replaces = [x for x in remaining_edits
                            if (x.startswith('I') or x.startswith('A'))]

        if len(inserts_replaces) == len(remaining_edits):
            return len(subword)
        else:
            return len(subword[: -(len(remaining_edits) - len(inserts_replaces))])

    def is_applicable(self, subword):
        _subword = subword.replace('##', '')
        char_edits = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|D\*|.', self.edit)
        char_edits_wo_append_merge = [e for e in char_edits if (not e.startswith('A') and
                                      not e.startswith('M') and not e.startswith('I'))]

        if self.edit == 'K' or self.edit.startswith('KA'):
            return True

        if len(_subword) < len(char_edits_wo_append_merge):
            return False

        idx = 0

        for i, edit in enumerate(char_edits_wo_append_merge):
            if edit.startswith('R') or edit in ['K', 'D']:
                idx += 1

            elif edit in ['K*', 'D*']:
                if i == len(char_edits_wo_append_merge) - 1:
                    idx += len(_subword[idx:])
                    break

                idx = len(_subword[: -len(char_edits_wo_append_merge[i + 1:])])

        if idx < len(_subword):
            return False

        return True

    def to_dict(self):
        return {'subword': self.subword, 'raw_subword': self.raw_subword, 'edit': self.edit}

    def to_json_str(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Edit compression helpers (from edits/edit.py)
# ---------------------------------------------------------------------------

def compress_edit(edit):
    grouped_edits = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D+|K+|.', edit)
    grouped_edits = compress_insertions(grouped_edits)
    return ''.join(grouped_edits)


def compress_insertions(edits):
    _edits = []
    insertions = ''
    for edit in edits:
        if edit.startswith('I_'):
            insertions += re.sub(r'I_\[(.*?)\]', r'\1', edit)
        else:
            if insertions:
                _edits.append(f'I_[{insertions}]')
                insertions = ''
            _edits.append(edit)

    if insertions:
        _edits.append(f'I_[{insertions}]')

    return _edits


# ---------------------------------------------------------------------------
# Rewrite functions (from gec/tag.py)
# ---------------------------------------------------------------------------

def resolve_merges(sent, edits):
    _sent = []
    assert len(sent) == len(edits)
    for subword, edit in zip(sent, edits):
        if edit.startswith('M'):
            if len(_sent) > 0:
                _sent[-1] = _sent[-1] + subword
            else:
                _sent.append(subword)
        else:
            _sent.append(subword)
    return _sent


def detokenize_sent(sent):
    detokenized = []
    for subword in sent:
        if subword.startswith('##'):
            detokenized[-1] = detokenized[-1] + subword.replace('##', '')
        else:
            detokenized.append(subword)
    return ' '.join(detokenized)


def rewrite(subwords, edits):
    assert len(subwords) == len(edits)

    rewritten_sents = []
    rewritten_sents_merge = []
    non_app_edits = []

    for i, (sent_subwords, sent_edits) in enumerate(zip(subwords, edits)):
        if len(sent_subwords) != len(sent_edits):
            assert len(sent_subwords) > len(sent_edits)
            sent_edits += ['K*'] * (len(sent_subwords) - len(sent_edits))

        rewritten_sent = []

        for subword, edit in zip(sent_subwords, sent_edits):
            edit_obj = SubwordEdit(subword=subword, raw_subword=subword, edit=edit)

            if edit_obj.is_applicable(subword):
                rewritten_subword = edit_obj.apply(subword)
                rewritten_sent.append(rewritten_subword)
            else:
                non_app_edits.append({'subword': subword, 'edit': edit_obj.to_json_str()})
                rewritten_sent.append(subword)

        rewritten_sents_merge.append(resolve_merges(rewritten_sent, sent_edits))
        rewritten_sents.append(rewritten_sent)

    detok_rewritten_sents = [detokenize_sent(sent) for sent in rewritten_sents_merge]
    return detok_rewritten_sents, rewritten_sents, non_app_edits


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_sweet_model(model_name=DEFAULT_MODEL_NAME):
    """
    Load a SWEET BertForTokenClassification model and tokenizer from HuggingFace.

    Downloads ~440MB on first run.

    Returns:
        (model, tokenizer) tuple
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer


def predict(model, tokenizer, words, decode_iter=2):
    """
    Run SWEET inference on a list of words (a pre-split sentence).

    Args:
        model: BertForTokenClassification model
        tokenizer: BertTokenizer
        words: list of str -- the sentence split into words (sentence.split())
        decode_iter: number of predict-rewrite iterations (default 2)

    Returns:
        Corrected sentence as a string
    """
    text = words
    device = next(model.parameters()).device

    for _ in range(decode_iter):
        tokenized = tokenizer(text, return_tensors="pt", is_split_into_words=True)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            logits = model(**tokenized).logits
            preds = F.softmax(logits.squeeze(), dim=-1)
            preds = torch.argmax(preds, dim=-1).cpu().numpy()
            edits = [model.config.id2label[p] for p in preds[1:-1]]

        subwords = tokenizer.convert_ids_to_tokens(tokenized['input_ids'][0][1:-1].cpu())
        text = rewrite(subwords=[subwords], edits=[edits])[0][0]
        text = text.split()

    return ' '.join(text)


if __name__ == "__main__":
    print("Loading SWEET model...")
    model, tok = load_sweet_model()
    test_input = "يجب الإهتمام ب الصحه".split()
    print(f"Input:  {' '.join(test_input)}")
    result = predict(model, tok, test_input)
    print(f"Output: {result}")
