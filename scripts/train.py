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
import optparse
import pandas as pd
import tqdm
import numpy as np
import joblib

sys.path.append('')


def _train(params):
    print("Buidling model")
    import joblib
    import numpy as np

    output_base = params.input_base
    print("Loading")
    x_train = joblib.load('{0}.trainX'.format(output_base))
    y_train = joblib.load('{0}.trainY'.format(output_base))
    x_dev = joblib.load('{0}.devX'.format(output_base))
    y_dev = joblib.load('{0}.devY'.format(output_base))

    from sklearn.feature_extraction.text import CountVectorizer

    print("\t::Fitting vectorizer")
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(x_train)
    y = np.array(y_train)
    joblib.dump(vectorizer, '{0}.vectorizer'.format(params.output_base))
    x_dev = vectorizer.transform(x_dev)
    y_dev = np.array(y_dev)

    print("Fitting")
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(verbose=2, n_jobs=16, n_estimators=50)
    clf.fit(x, y)
    joblib.dump(clf, '{0}.classifier2'.format(params.output_base))

    y_pred = clf.predict(x_dev)
    from sklearn.metrics import f1_score

    print("F1 = {0}".format(f1_score(y_dev, y_pred)))
    from sklearn.metrics import confusion_matrix

    print("Confusion matrix:")
    print(confusion_matrix(y_dev, y_pred))

    print("\nDone")


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--input-base', action='store', dest='input_base', help='base path for input files (train+dev)')
    parser.add_option('--output-base', action='store', dest='output_base', help='location of the output model')
    (params, _) = parser.parse_args(sys.argv)

    if params.input_file and params.output_base and params.platform:
        _train(params)
    else:
        parser.print_help()
