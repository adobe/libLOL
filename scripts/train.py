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
import optparse
import pandas as pd
import tqdm
import numpy as np
import joblib

sys.path.append('')


def _auto_split(x_d, y_d, commands):
    x0 = []
    x1 = []
    command0 = []
    command1 = []
    for x, y, c in zip(x_d, y_d, commands):
        if y == 1:
            x1.append(x)
            command1.append(c)
        else:
            x0.append(x)
            command0.append(c)

    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    command_dev = []
    cnt = 0
    for x, command in zip(x0, command0):
        cnt += 1
        if cnt % 10 == 0:
            x_dev.append(x)
            y_dev.append(0)
            command_dev.append(command)
        else:
            x_train.append(x)
            y_train.append(0)

    for x, command in zip(x1, command1):
        cnt += 1
        if cnt % 10 == 0:
            x_dev.append(x)
            y_dev.append(1)
            command_dev.append(command)
        else:
            x_train.append(x)
            y_train.append(1)

    return x_train, y_train, x_dev, y_dev, command_dev


def _train(params):
    print("Buidling model")
    from lol.model.utils import LinuxFeatureExtraction, WindowsFeatureExtraction, ParallelFTE
    if params.platform == 'linux':
        fte = LinuxFeatureExtraction()
    else:
        fte = WindowsFeatureExtraction()

    dataset = pd.read_csv(params.input_file)
    x = []
    y = []

    all_commands = dataset['commands'].tolist()
    commands = []
    tag = dataset['tag'].tolist()
    del dataset
    n_batch = len(all_commands) // params.batch_size
    if len(all_commands) % params.batch_size != 0:
        n_batch += 1

    pfte = ParallelFTE(fte)

    b_features = pfte(all_commands, n_jobs=params.batch_size)

    for features, command, ctag in zip(b_features, all_commands, tag):
        if ctag == 'good':
            y.append(0)
        else:
            if len(features) < 3:
                continue
            y.append(1)
        commands.append(command)
        x.append(' '.join(features))
    # pgb = tqdm.tqdm(zip(dataset['commands'], dataset['tag']), desc='\t::Extracting features', ncols=160,
    #                 total=len(dataset['commands']))
    # for command, tag in pgb:
    #     features = fte(command, training=True)
    #     if tag == 'good':
    #         y.append(0)
    #     else:
    #         if len(features) < 3:  # don't use any negative examples with less than three tags - too many FPs
    #             continue
    #         y.append(1)
    #
    #     commands.append(command)
    #     x.append(' '.join(features))

    if params.auto_split:
        x_train, y_train, x_dev, y_dev, command_dev = _auto_split(x, y, commands)
    else:
        x_train = x
        y_train = y
        x_dev = x
        y_dev = y
        command_dev = commands

    from sklearn.feature_extraction.text import CountVectorizer
    print("\t::Fitting vectorizer")
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(x_train).toarray()
    y = np.array(y_train)
    print("\t::Fitting classifier")
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(verbose=True)
    clf.fit(x, y)
    joblib.dump(vectorizer, '{0}.vectorizer'.format(params.output_base))
    joblib.dump(clf, '{0}.classifier'.format(params.output_base))
    joblib.dump(x, '{0}.trainX'.format(params.output_base))
    joblib.dump(y, '{0}.trainY'.format(params.output_base))

    if x_dev is not None:
        print("\t::Evaluating")
        x_dev = vectorizer.transform(x_dev).toarray()
        y_pred = clf.predict(x_dev)
        from sklearn.metrics import f1_score
        print("F1 = {0}".format(f1_score(y_dev, y_pred)))
        from sklearn.metrics import confusion_matrix
        print("Confusion matrix:")
        print(confusion_matrix(y_dev, y_pred))

        f_err = open('train.err', 'w')
        f_log = open('train.log', 'w')
        for y_t, y_p, cmd in zip(y_dev, y_pred, command_dev):
            if y_t != y_p:
                f_err.write('{0} {1} {2}\n'.format(y_t, y_p, cmd))
            if y_t == 1 or (y_t != y_p):
                f_log.write('{0} {1} {2}\n'.format(y_t, y_p, cmd))
        f_err.close()
        f_log.close()
        print("Verbose results stored in train.err and train.log")

    print("\nDone")


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-file', action='store', dest='input_file', help='location of the input file')
    parser.add_option('--output-base', action='store', dest='output_base', help='location of the output model')
    parser.add_option('--auto-split', action='store_true', dest='auto_split', help='location of the output model')
    parser.add_option('--platform', action='store', dest='platform', help='what platform to train for: linux, windows',
                      choices=['linux', 'windows'])
    parser.add_option('--num-jobs', action='store', dest='batch_size', help='default=16', type='int', default=16)
    parser.add_option('--classifier', action='store', dest='classifier', choices=['MLP', 'RF', 'GP'], default='MLP',
                      help='Classifier to use: MLP = MultiLayerPerceptron, RF = RandomForest, GP = Gaussian Process')
    (params, _) = parser.parse_args(sys.argv)

    if params.input_file and params.output_base and params.platform:
        _train(params)
    else:
        parser.print_help()
