---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

### Environment

If an issue occurs it is helpful to provide several information: Please *replace* these information below:

|info| output |
|---|---|
|os | `uname -a` |
|c++ version | `c++ --version` |
|TF version | `python -c "import tensorflow as tf; print(tf.__version__);"` |
|TF is working | `python -c "import tensorflow as tf; sess=tf.InteractiveSession()"` works |

You might run the [tf_info.sh script](https://gist.github.com/PatWie/c04f4d6a61ac201b8afa2c59fe9ba586) in your terminal to gather these information and manually anonymize the output when necessary before pasting it here.

### Issue

A clear and concise description of what the bug is will be helpful to resolve issues.

**Context:**
Please explain the conditions which led you to write this issue, eg. "writing a custom op" or "doing inference on device given a keras model".

**Reproduce:**
Please explain what you did (exactly), eg., files you modified, or, commands you entered so that I can reproduce if possible.

**Output**
Please explain what you get as an result or output from the *reproduce*-steps above, eg. "I get the output/error message ... during the compilation". Please provide the exact output and no interpretation.

**Expectation**
Please explain what you expected to get, eg. "the inference code x should produce y".

**Investigation**
If possible, please explain, which steps you did to investigate this issues. What are your findings.
