#
# Copyright (c) 2021 Adobe Systems Incorporated. All rights reserved.
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
import sys

sys.path.append('')
from lol.model.utils import LinuxFeatureExtraction, WindowsFeatureExtraction
from enum import Enum
from typing import Optional, Union, List
import pkg_resources
import joblib


class PlatformType(Enum):
    """
    Supported platforms
    """
    LINUX: str = 'linux'
    WINDOWS: str = 'windows'


class LOLC:
    def __init__(self, platform: PlatformType, model_base: Optional[str] = None):
        if model_base is None:
            base_path = pkg_resources.resource_filename(__name__, 'data/bash.bad.filtered') \
                .replace('bash.bad.filtered', '')
        else:
            base_path = model_base
       
        if platform == PlatformType.LINUX:
            if model_base is None:
                base_path = '{0}/bash_huge'.format(base_path)
            fte = LinuxFeatureExtraction(base_path=base_path)
            vectorizer_path = '{0}.vectorizer'.format(base_path)
            classifier_path = '{0}.classifier2'.format(base_path)
        elif platform == PlatformType.WINDOWS:
            if model_base is None:
                base_path = '{0}/cmd_huge'.format(base_path)
            fte = WindowsFeatureExtraction(base_path=base_path)
            vectorizer_path = '{0}.vectorizer'.format(base_path)
            classifier_path = '{0}.classifier2'.format(base_path)
        else:
            raise Exception("Unknown platform type")

        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(classifier_path)
        self._fte = fte
        self._vectorizer = vectorizer
        self._classifier = classifier

    def __call__(self, command: Union[str, List[str]]) -> (Union[str, List[str]], Union[List[str], List[List[str]]]):
        if isinstance(command, str):
            commands = [command]
        else:
            commands = command

        list_class, list_tags = self._batched_process(commands)

        if isinstance(command, str):
            return list_class[0], list_tags[0]
        else:
            return list_class, list_tags

    def _batched_process(self, commands):
        tags = []
        status = []
        batch_feats = []
        batch_lines = []
        batch_size = 1000
        for line in commands:
            line = str(line).strip()
            feats = self._fte(line)
            batch_feats.append(feats)
            batch_lines.append(line)
            if len(batch_feats) == batch_size:
                self._process_batch(batch_feats, batch_lines, tags, status)
                batch_feats = []
                batch_lines = []

        if len(batch_feats) != 0:
            self._process_batch(batch_feats, batch_lines, tags, status)

        return status, tags

    def _process_batch(self, batch_feats, batch_lines, tags, status):
        new_batch = [' '.join(feats) for feats in batch_feats]
        x = self._vectorizer.transform(new_batch).toarray()
        y = self._classifier.predict(x)
        for feats, line, label in zip(batch_feats, batch_lines, y):
            if len(feats) < 3 and 'LOOKS_LIKE_KNOWN_LOL' not in feats:
                tags.append('')
                status.append('NEUTRAL')
            else:
                features = ' '.join(feats)
                tags.append(features)
                print(label)
                if label == 1 or 'LOOKS_LIKE_KNOWN_LOL' in feats:
                    status.append('BAD')
                else:
                    status.append('GOOD')
