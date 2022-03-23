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

import sys

sys.path.append('')
import optparse
import joblib


def _process_batch(batch_feats, batch_lines, vectorizer, clf, tags, status):
    new_batch = [' '.join(feats) for feats in batch_feats]
    x = vectorizer.transform(new_batch).toarray()
    y = clf.predict(x)
    for feats, line, label in zip(batch_feats, batch_lines, y):
        if len(feats) < 3 and 'LOOKS_LIKE_KNOWN_LOL' not in feats:
            label = 0
            tags.append('')
            if params.verbose:
                features = ' '.join(feats)
                print("==========\n" + line + "\n\nLabels: " + features + "\nNO VERDICT\n=========\n\n")
        else:
            features = ' '.join(feats)
            tags.append(features)
            # x = vectorizer.transform([features]).toarray()
            # # label = clf.predict(x)[0]
        if label == 1 or 'LOOKS_LIKE_KNOWN_LOL' in feats:
            status.append('bad')
            print("==========\n" + line + "\n\nLabels: " + features + "\nBAD\n=========\n\n")
            sys.stdout.flush()
        else:
            status.append('good')
            if params.verbose:
                features = ' '.join(feats)
                print("==========\n" + line + "\n\nLabels: " + features + "\nNEUTRAL\n=========\n\n")


def _test(params):
    from lol.model.utils import LinuxFeatureExtraction, WindowsFeatureExtraction
    if params.platform == 'linux':
        fte = LinuxFeatureExtraction()
    else:
        fte = WindowsFeatureExtraction()

    vectorizer = joblib.load('{0}.vectorizer'.format(params.output_base))
    clf = joblib.load('{0}.classifier2'.format(params.output_base))
    if not params.test_file:
        while True:
            sys.stdout.write(">>> ")
            sys.stdout.flush()
            text = input()
            if text == "exit":
                return
            feats = fte(text)
            features = ' '.join(feats)
            x = vectorizer.transform([features]).toarray()
            label = clf.predict(x)[0]
            if label == 0 and 'LOOKS_LIKE_KNOWN_LOL' not in feats:
                label = "GOOD"
            else:
                label = "BAD"
            print("Extracted features: " + features)
            print("Prediction: {0}\n\n".format(label))
    else:
        if params.use_csv:
            import pandas as pd
            data = pd.read_csv(params.test_file)
            lines = data['command']
        else:
            lines = open(params.test_file).readlines()
        print("Bad commands")

        tags = []
        status = []
        batch_feats = []
        batch_lines = []
        batch_size = 1000
        for line in lines:
            line = str(line).strip()
            feats = fte(line)
            batch_feats.append(feats)
            batch_lines.append(line)
            if len(batch_feats) == batch_size:
                _process_batch(batch_feats, batch_lines, vectorizer, clf, tags, status)
                batch_feats = []
                batch_lines = []

        if len(batch_feats) != 0:
            _process_batch(batch_feats, batch_lines, vectorizer, clf, tags, status)

        if params.output_file:
            data['tags'] = tags
            data['status'] = status
            bad_data = data[data['status'] == 'bad']
            bad_data.to_csv(params.output_file)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--output-base', action='store', dest='output_base', help='location of the output model')
    parser.add_option('--test-file', action='store', dest='test_file', help='location of the test file (optional)')
    parser.add_option('--use-csv', action='store_true', dest='use_csv', help='is data in csv format')
    parser.add_option('--csv-column', action='store', dest='csv_column', help='default(command)', default='command')
    parser.add_option('--output-file', action='store', dest='output_file', help='where to store the results (optional)')
    parser.add_option('--platform', action='store', dest='platform', help='what platform to train for: linux, windows',
                      choices=['linux', 'windows'])
    parser.add_option('--verbose', action='store_true', dest='verbose')
    (params, _) = parser.parse_args(sys.argv)

    if params.output_base and params.platform:
        _test(params)
    else:
        parser.print_help()
