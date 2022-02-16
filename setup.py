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

import setuptools


def parse_requirements(filename, session=None):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lolc",
    version="0.1.0.4",
    author="Multiple authors",
    author_email="tiberiu44@gmail.com",
    description="Python module for detecting password, api keys hashes and any other string that resembles a randomly generated character sequence.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adobe/lolc",
    packages=setuptools.find_packages(),
    install_requires=parse_requirements('requirements.txt', session=False),
    classifiers=(
        "Programming Language :: Python :: 3.0",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    data_files=[
        ('lol', ['data/bash_huge.vectorizer',
              'data/bash_huge.classifier2',
              'data/bash_huge.known',
              'data/cmd_huge.known',
              'data/cmd_huge.vectorizer',
              'data/cmd_huge.classifier2'])
    ],
    package_data={
        '': ['data/bash_huge.vectorizer', 'data/bash_huge.classifier2', 'data/bash_huge.known', 'data/cmd_huge.known',
             'data/cmd_huge.vectorizer', 'data/cmd_huge.classifier2']
    },
    zip_safe=False
)
