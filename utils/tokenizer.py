from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import json
import logging
import os
import regex as re
from io import open
import pickle
import six

import collections
import logging
import os
import unicodedata
from io import open

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_bert import BertTokenizer

class ChineseEnglishTokenizer(PreTrainedTokenizer):
    """
    Chinese-English BPE tokenizer. Peculiarities:
        - Byte-level Byte-Pair-Encoding
        - Requires a space to start the input string => the encoding methods should be called with the
          ``add_prefix_space`` flag set to ``True``.
          Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
          the absence of a space at the beginning of a string: `tokenizer.decode(tokenizer.encode("Hello")) = " Hello"`
    """

    def __init__(self, gpt2_en_tokenizer, bert_cn_tokenizer,
                 unk_token="<|endoftext|>", bos_token="<|endoftext|>", eos_token="<|endoftext|>", **kwargs):
        super(ChineseEnglishTokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.gpt2_en_tokenizer = gpt2_en_tokenizer
        self.bert_cn_tokenizer = bert_cn_tokenizer
        self.bert_cn_offset = len(self.gpt2_en_tokenizer.encoder)

    @property
    def vocab_size(self):
        return len(self.gpt2_en_tokenizer.encoder) + len(self.bert_cn_tokenizer.vocab)

    def encode_plus(self,
                    text,
                    text_pair=None,
                    add_special_tokens=True,
                    max_length=None,
                    stride=0,
                    truncation_strategy='longest_first',
                    return_tensors=None,
                    **kwargs):
        """
        Returns a dictionary containing the encoded sequence or sequence pair and additional informations:
        the mask for sequence classification and the overflowing elements if a ``max_length`` is specified.

        Args:
            text: The first sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method)
            text_pair: Optional second sequence to be encoded. This can be a string, a list of strings (tokenized
                string using the `tokenize` method) or a list of integers (tokenized string ids using the
                `convert_tokens_to_ids` method)
            add_special_tokens: if set to ``True``, the sequences will be encoded with the special tokens relative
                to their model.
            max_length: if set to a number, will limit the total sequence returned so that it has a maximum length.
                If there are overflowing tokens, those will be added to the returned dictionary
            stride: if set to a number along with max_length, the overflowing tokens returned will contain some tokens
                from the main sequence returned. The value of this argument defines the number of additional tokens.
            truncation_strategy: string selected in the following options:
                - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                    starting from the longest one at each token (when there is a pair of input sequences)
                - 'only_first': Only truncate the first sequence
                - 'only_second': Only truncate the second sequence
                - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
            return_tensors: (optional) can be set to 'tf' or 'pt' to return respectively TensorFlow tf.constant
                or PyTorch torch.Tensor instead of a list of python integers.
            **kwargs: passed to the `self.tokenize()` method
        """

        def get_input_ids(text):
            if isinstance(text, six.string_types):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], six.string_types):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError("Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers.")

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(first_ids,
                                      pair_ids=second_ids,
                                      max_length=max_length,
                                      add_special_tokens=add_special_tokens,
                                      stride=stride,
                                      truncation_strategy=truncation_strategy,
                                      return_tensors=return_tensors)    

    def _tokenize_chinese_chars(self,text):
        output = []
        for char in text:
            cp = ord(char)
            if self.bert_cn_tokenizer.basic_tokenizer._is_chinese_char(cp):
                output.append(' ')
                output.append(char)
                output.append(' ')
            else:
                output.append(char)
        return "".join(output)    
    
    def _tokenize(self, text, **kwargs):
        # Preprocessing for chinese character
        text = self.bert_cn_tokenizer.basic_tokenizer._clean_text(text) 
        text = self._tokenize_chinese_chars(text)
        while '  ' in text:
            text = text.replace('  ',' ')
            
        bpe_tokens = []
        for token in re.findall(self.gpt2_en_tokenizer.pat, text):
            is_chinese_char = self.bert_cn_tokenizer.basic_tokenizer._is_chinese_char(ord(token[-1])) if len(token) == 2 else False
            if is_chinese_char:
                token = self.bert_cn_tokenizer.basic_tokenizer._run_split_on_punc(token[-1])
                bpe_tokens.extend(token)
            else:
                if sys.version_info[0] == 2:
                    # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    token = ''.join(self.gpt2_en_tokenizer.byte_encoder[ord(b)] for b in token) 
                else:
                    # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    token = ''.join(self.gpt2_en_tokenizer.byte_encoder[b] for b in token.encode('utf-8')) 
                bpe_tokens.extend(bpe_token for bpe_token in self.gpt2_en_tokenizer.bpe(token).split(' '))
        return bpe_tokens
    
    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        is_chinese_char = self.bert_cn_tokenizer.basic_tokenizer._is_chinese_char(ord(token)) if len(token) == 1 else False
        unk_id = self.gpt2_en_tokenizer.encoder.get(self.unk_token)
        if is_chinese_char:
            token_id = self.bert_cn_tokenizer.vocab.get(token, None)
            if token_id is None:
                return unk_id
            else:
                return token_id + self.bert_cn_offset
        else:
            return self.gpt2_en_tokenizer.encoder.get(token, unk_id)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        if index >= self.bert_cn_offset:
            return self.bert_cn_tokenizer.ids_to_tokens.get(index - self.bert_cn_offset, self.unk_token)
        else:
            return self.gpt2_en_tokenizer.decoder.get(index)
                                                     
    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        text = ''.join(tokens)
        bytes_rep = [self.gpt2_en_tokenizer.byte_decoder[c] if c in self.gpt2_en_tokenizer.byte_decoder else c for c in text]
        s_idx = 0
        e_idx = 0
        i, j = 0, 0
        text_list = []
        while i < len(bytes_rep):
            if isinstance(bytes_rep[i], int):
                j = i + 1
                while j < len(bytes_rep):
                    if isinstance(bytes_rep[j], str):
                        text_list.append(bytearray(bytes_rep[i:j]).decode('utf-8', errors=self.gpt2_en_tokenizer.errors))
                        i = j
                        break
                    else:
                        j += 1

                if i != j: # We reach end of string without any chinese character
                    text_list.append(bytearray(bytes_rep[i:]).decode('utf-8', errors=self.gpt2_en_tokenizer.errors))
                    break
            else:
                j = i + 1
                while j < len(bytes_rep):
                    if isinstance(bytes_rep[j], int):
                        text_list.append(''.join(bytes_rep[i:j]))
                        i = j
                        break
                    else:
                        j += 1

                if i != j: # We reach end of string without any latin character
                    text_list.append(''.join(bytes_rep[i:]))
                    break
        return ' '.join(text_list)
    
if __name__ == '__main__':
    gpt2_en_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    bert_cn_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    cn_en_tokenizer = ChineseEnglishTokenizer(gpt2_en_tokenizer, bert_cn_tokenizer)
    
    texts = [
        'pokemon你好吗 我好吗digimon gokemon',
        '六六六六六六六六',
        '我爱你, 你爱我吗?',
        '我，要。吃？饭！！',
        'I come from 北京 , I wanna go to 香港',
        '666 I come from 深圳 , I wanna eat 角质 at chinese restaurant',
        "我 的 帅 管家",
        "就 那种 typical 的 那 种 偶像 剧",
        "没有 没 有 爸爸 爸爸 爸爸 不 要 跟 家里 在一起 了 因为",
        "它 不可以 把",
        "还 有 假 睫毛 瞳孔 放大 片 加 假 睫毛 然后 就 可以 把 一 个 丑 女 变 美女",
        "你 讲 你 讲 你 讲",
        "三倍",
        "那个 胸部 也 是 我 爸 那 时候 看 那个",
        "没有 无所谓 我 只是 说 它 可是 把 它 挤 出来 而已",
        "没有 我 本来 我 刚才 有 东西 要 讲 的 你 插 我 的 话",
        "我 刚刚 讲到 差 很大",
        "那个 我 看 台湾 节目 一 大 堆",
        "就 她们 卸妆 然后 整个 变得 超 夸张 welcome 也是",
        "他 本来 爸爸 本来 有 未婚妻 ok 然后 他 不要 跟 他 未婚妻 结婚 他 喜欢 另外 一 个 女孩子",
        "welcome 它 每次 找 一 个 卸妆 的 美女",
        "welcome 外星人 是 一 个 台湾 的 综 艺 节目",
        "然后 我 要 讲 的 是 他们 不 是 每 次 都 找 乱七八糟 的 人 以前 是 找 真的 那 种 怪怪 的 人 可是 现在 主题 越 做 越 普遍 了 就是 做 好像 我 猜 的 那 种 人 了",
        "I wanna be the very best like noone ever was!!"
    ]
    for text in texts:
        enc = cn_en_tokenizer.encode(text)
        dec = cn_en_tokenizer.decode(enc)
        print('==== text enc dec ====')
        print(text)
        print('====')
        print(enc)
        print('====')
        print(dec)
        print('====')
        print(cn_en_tokenizer.gpt2_en_tokenizer.encoder['<|endoftext|>'], cn_en_tokenizer.eos_token_id)
        print('|'.join(list(cn_en_tokenizer.gpt2_en_tokenizer.encoder.keys())[100:200]))