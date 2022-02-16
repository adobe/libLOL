#
# Copyright (c) 2022 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime

import re2 as re
import sys
import collections
import math
from abc import abstractmethod
import shlex
import threading
from multiprocessing import Process

sys.path.append('')
from ipaddress import ip_address


class FeatureExtraction:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, command: str, training=False) -> [str]:
        pass


def cmdline_split(s, platform=0):
    """Multi-platform variant of shlex.split() for command-line splitting.
    For use with subprocess, for argv injection etc. Using fast REGEX.

    platform: 'this' = auto from current platform;
              1 = POSIX;
              0 = Windows/CMD
              (other values reserved)
    """
    if platform == 'this':
        platform = (sys.platform != 'win32')
    if platform == 1:
        RE_CMD_LEX = r'''"((?:\\["\\]|[^"])*)"|'([^']*)'|(\\.)|(&&?|\|\|?|\d?\>|[<])|([^\s'"\\&|<>]+)|(\s+)|(.)'''
    elif platform == 0:
        RE_CMD_LEX = r'''"((?:""|\\["\\]|[^"])*)"?()|(\\\\(?=\\*")|\\")|(&&?|\|\|?|\d?>|[<])|([^\s"&|<>]+)|(\s+)|(.)'''
    else:
        raise AssertionError('unkown platform %r' % platform)

    args = []
    accu = None  # collects pieces of one arg
    for qs, qss, esc, pipe, word, white, fail in re.findall(RE_CMD_LEX, s):
        if word:
            pass  # most frequent
        elif esc:
            word = esc[1]
        elif white or pipe:
            if accu is not None:
                args.append(accu)
            if pipe:
                args.append(pipe)
            accu = None
            continue
        elif fail:
            raise ValueError("invalid or incomplete shell string")
        elif qs:
            word = qs.replace('\\"', '"').replace('\\\\', '\\')
            if platform == 0:
                word = word.replace('""', '"')
        else:
            word = qss  # may be even empty; must be last

        accu = (accu or '') + word

    if accu is not None:
        args.append(accu)

    return args


def cmdline_split_all(s: str, platform=0):
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    c_toks = [s]
    i_toks = cmdline_split(s, platform=platform)
    while c_toks != i_toks:
        i_toks = c_toks
        c_toks = []
        for tok in i_toks:
            if tok.startswith('"') and tok.endswith('"'):
                tok = tok[1:-1]
            tmp = cmdline_split(tok, platform=platform)
            for nt in tmp:
                c_toks.append(nt)

    return c_toks


def posix_split_all(s: str, platform=1):
    text = s
    i_toks = [text]
    c_toks = list(shlex.shlex(text, punctuation_chars=True, posix=True))
    while c_toks != i_toks:
        i_toks = c_toks
        c_toks = []
        for tok in i_toks:
            tmp = list(shlex.shlex(tok, punctuation_chars=True, posix=True))
            for nt in tmp:
                c_toks.append(nt)

    return c_toks


_re_ipv4 = r"[^\w^\d]((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)[^\w^\d]"


def _match_ips(text):
    try:
        ips = re.finditer(_re_ipv4, text)
        labels = {}
        for ip in ips:
            try:
                ipt = ip_address(''.join(ch for ch in str(ip.group()) if ch.isalnum() or ch == "."))
                if ipt.is_link_local:
                    resp = "link_local"
                elif ipt.is_loopback:
                    resp = "loopback"
                elif ipt.is_multicast:
                    resp = "multicast"
                elif ipt.is_reserved:
                    resp = "reserved"
                elif ipt.is_unspecified:
                    resp = "unspecified"
                elif ipt.is_private:
                    resp = "private"
                else:
                    resp = "public"

                resp = "IP_{0}".format(resp).upper()
                if resp not in labels:
                    labels[resp] = 1
            except:
                pass
    except:
        labels = {}
    return [label for label in labels]


class SimilarityLabelGenerator:
    def __init__(self, filename: str, platform='linux'):
        #print(filename)
        self._platform = platform
        lines = open(filename).readlines()
        self._strings = []
        for s in lines:
            # s = s[:min(150, len(s))]
            s = s.strip()
            if len(s) > 100 or len(s) < 5:
                continue
            try:
                if platform == 'linux':
                    cs = posix_split_all(s)
                else:
                    cs = cmdline_split_all(s)
                self._strings.append(cs)
            except:
                print("Invalid command: ")
                print(s + '\n\n')
                sys.stdout.flush()

    def __call__(self, string, threshold=0.5) -> bool:
        string = str(string)
        if len(string) > 100 or len(string) < 6:
            return False

        import warnings
        from nltk.translate.bleu_score import sentence_bleu
        try:
            string = re.sub("[0-9]+", "0", string)
            # candidate = list(shlex.shlex(string, punctuation_chars=True, posix=True))  # [c for c in string]
            if self._platform == 'linux':
                candidate = posix_split_all(string)
            else:
                candidate = cmdline_split_all(string)
        except:
            print("Invalid input command: " + string)
            sys.stdout.flush()
            return False
        if len(candidate) < 4:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sc in self._strings:
                score = sentence_bleu([sc], candidate, weights=(0.25, 0.25, 0.25, 0.25))
                if score > threshold:
                    return True
        return False


class RegexLabelGenerator:
    def __init__(self, regex_dict):
        self._regex_dict = regex_dict
        self._compiled_regex = [re.compile(regex, re.MULTILINE) for regex in
                                regex_dict]  # , max_mem=1024 * 1024 * 1024)

    def __call__(self, text):
        text = ' {0} '.format(text)
        labels = []
        for cr in self._compiled_regex:
            matches = re.finditer(cr, ' ' + text)
            for matchNum, match in enumerate(matches, start=1):
                for elem in match.groupdict():
                    if match.groupdict()[elem] is not None:
                        labels.append(str(elem))

        return labels


class KeywordLabelGenerator:
    def __init__(self, prefix: str, keywords: [str], tokenize: bool, platform='linux'):
        self._prefix = prefix
        self._keywords = {k.lower(): 0 for k in keywords}  # for fast indexing
        self._tokenize = tokenize
        self._platform = platform

    def __call__(self, text: str):
        text = str(text)
        labels = {}
        if self._tokenize:
            if self._platform == 'linux':
                try:
                    toks = posix_split_all(text)
                except:
                    toks = []
            else:
                toks = cmdline_split_all(text, platform=0)
            for tok in toks:
                tok = tok.lower()
                if tok in self._keywords:
                    label = '{0}_{1}'.format(self._prefix, tok).upper()
                    if label not in labels:
                        labels[label] = 0
        else:
            for keyword in self._keywords:
                if keyword in text:
                    label = '{0}_{1}'.format(self._prefix, keyword).upper()
                    if label not in labels:
                        labels[label] = 0

        return [label for label in labels]


class WindowsFeatureExtraction(FeatureExtraction):
    def __init__(self, base_path: str):
        super().__init__()
        from lol.model.constants import WINDOWS_PATHS, WINDOWS_COMMANDS, WINDOWS_KEYWORDS, WINDOWS_REGEX_LIST
        self._paths = KeywordLabelGenerator("PATH", WINDOWS_PATHS, False)
        self._commands = KeywordLabelGenerator("COMMAND", WINDOWS_COMMANDS, True, platform='windows')
        self._keywords = KeywordLabelGenerator("KEYWORD", WINDOWS_KEYWORDS, True, platform='windows')
        if base_path is None:
            self._similarity = SimilarityLabelGenerator('data/cmd.bad.filtered', platform='linux')
        else:
            self._similarity = SimilarityLabelGenerator('{0}.known'.format(base_path), platform='linux')
        self._regex = RegexLabelGenerator(WINDOWS_REGEX_LIST)

    def __call__(self, command: str, training=False) -> [str]:
        labels = _match_ips(command)
        l_tmp = self._paths(command)
        for lab in l_tmp:
            labels.append(lab)
        l_tmp = self._commands(command)
        for lab in l_tmp:
            labels.append(lab)
        l_tmp = self._keywords(command)
        for lab in l_tmp:
            labels.append(lab)
        l_tmp = self._regex(command)
        for lab in l_tmp:
            labels.append(lab)

        if not training and self._similarity(command):
            labels.append('LOOKS_LIKE_KNOWN_LOL')
        labels = list(set(labels))
        return labels


class LinuxFeatureExtraction(FeatureExtraction):
    def __init__(self, base_path=None):
        super().__init__()
        from lol.model.constants import LINUX_PATHS, LINUX_COMMANDS, LINUX_KEYWORDS, LINUX_REGEX_LIST
        self._paths = KeywordLabelGenerator("PATH", LINUX_PATHS, False)
        self._commands = KeywordLabelGenerator("COMMAND", LINUX_COMMANDS, True)
        self._keywords = KeywordLabelGenerator("KEYWORD", LINUX_KEYWORDS, True)
        if base_path is None:
            self._similarity = SimilarityLabelGenerator('data/bash.bad.filtered', platform='linux')
        else:
            self._similarity = SimilarityLabelGenerator('{0}.known'.format(base_path), platform='linux')
        self._regex = RegexLabelGenerator(LINUX_REGEX_LIST)

    def __call__(self, command: str, training=False) -> [str]:
        labels = _match_ips(command)
        l_tmp = self._paths(command)
        for lab in l_tmp:
            labels.append(lab)
        l_tmp = self._commands(command)
        for lab in l_tmp:
            labels.append(lab)
        l_tmp = self._keywords(command)
        for lab in l_tmp:
            labels.append(lab)
        l_tmp = self._regex(command)
        for lab in l_tmp:
            labels.append(lab)

        if not training and self._similarity(command):
            labels.append('LOOKS_LIKE_KNOWN_LOL')
        labels = list(set(labels))
        return labels


class CustomBLEU:
    def __init__(self):
        pass

    def _get_ngrams(self, segment, min_order, max_order):
        """Extracts all n-grams upto a given maximum order from an input segment.
        Args:
          segment: text segment from which n-grams will be extracted.
          max_order: maximum length in tokens of the n-grams returned by this
              methods.
        Returns:
          The Counter containing all n-grams upto max_order in segment
          with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(min_order, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i + order])
                ngram_counts[ngram] += 1
        #  print ngram_counts
        return ngram_counts

    def compute_bleu(self, reference_corpus, translation_corpus, min_order=3, max_order=4,
                     smooth=False):
        """Computes BLEU score of translated segments against one or more references.
        Args:
          reference_corpus: list of lists of references for each translation. Each
              reference should be tokenized into a list of tokens.
          translation_corpus: list of translations to score. Each translation
              should be tokenized into a list of tokens.
          max_order: Maximum n-gram order to use when computing BLEU score.
          smooth: Whether or not to apply Lin et al. 2004 smoothing.
        Returns:
          3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
          precisions and brevity penalty.
        """
        matches_by_order = [0] * max_order
        possible_matches_by_order = [0] * max_order
        reference_length = 0
        translation_length = 0
        for (references, translation) in zip(reference_corpus,
                                             translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference, min_order, max_order)
            translation_ngram_counts = self._get_ngrams(translation, min_order, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = ((matches_by_order[i] + 1.) /
                                 (possible_matches_by_order[i] + 1.))
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (float(matches_by_order[i]) /
                                     possible_matches_by_order[i])
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)

        bleu = geo_mean * bp

        return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _print_stats(id, start, stop, count, total):
    span = stop - start
    if total == 0:
        total += 1
    proc = count * 100 / total
    left = 100 - proc
    if proc == 0:
        proc = 1
    eta = left * span / proc
    sys.stdout.write('Thread {0} elapsed {1} progress {2} eta {3}\n'.format(id, span, proc, eta))
    sys.stdout.flush()


def _thread_function(commands: [], id: int, fte: FeatureExtraction):
    sys.stdout.write('Thread {0} started\n'.format(id))
    sys.stdout.flush()
    # fte_list[id] = []  # feats
    current_list = []
    count = 0
    start = datetime.datetime.now()
    for command in commands:
        count += 1
        if count % 1000 == 0:
            stop = datetime.datetime.now()
            _print_stats(id, start, stop, count, len(commands))
        feats = fte(command)

        current_list.append(feats)
    import pickle
    f = open('f_{0}.pickle'.format(id), 'wb')
    pickle.dump(current_list, f)
    f.close()
    f = open('f_{0}.done'.format(id), 'w')
    f.write('done')
    f.close()
    stop = datetime.datetime.now()
    _print_stats(id, start, stop, count, len(commands))


class ParallelFTE:
    def __init__(self, fte: FeatureExtraction):
        self._fte = fte

    def __call__(self, commands: [], n_jobs=16):
        # fte = [[] for _ in range(len(commands))]
        import multiprocessing
        # manager = multiprocessing.Manager()
        # fte_dict = manager.dict()
        batch_size = len(commands) // n_jobs
        if len(commands) % n_jobs != 0:
            batch_size += 1

        for id in range(n_jobs):
            import os
            process_output_file = 'f_{0}.pickle'.format(id)
            process_check_file = 'f_{0}.done'.format(id)
            if os.path.exists(process_output_file):
                os.unlink(process_output_file)
            if os.path.exists(process_check_file):
                os.unlink(process_check_file)

        for batch in range(n_jobs):
            start = batch * batch_size
            stop = min(len(commands), start + batch_size)
            thread_list = []
            print(batch, start, stop)
            thread = Process(target=_thread_function, args=(commands[start:stop], batch, self._fte))
            thread_list.append(thread)
            thread.start()

        for thread in thread_list:
            thread.join()

        # join seems to fail in some cases
        all_done = False
        while not all_done:
            import time
            import os
            time.sleep(1)
            all_done = True
            for id in range(n_jobs):
                process_check_file = 'f_{0}.done'.format(id)
                if not os.path.exists(process_check_file):
                    all_done = False

        fte_list = []
        for id in range(n_jobs):
            import pickle
            import os
            process_output_file = 'f_{0}.pickle'.format(id)
            fte_set = pickle.load(open(process_output_file, 'rb'))  # fte_dict[id]
            os.unlink(process_output_file)
            process_check_file = 'f_{0}.done'.format(id)
            os.unlink(process_check_file)
            for feats in fte_set:
                fte_list.append(feats)

        return fte_list


if __name__ == '__main__':
    print(_match_ips("nc 127.0.0.1 89.38.230.11 10.0.2.3 255.0.0.0 224.0.0.0"))
