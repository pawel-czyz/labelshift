![build](https://github.com/labelshift/labelshift/actions/workflows/build/badge.svg)

# Label Shift

Python library for *label shift* (known as prior probability shift, target shift) and *quantification* (estimating the class prevalences in an unlabeled data set under the prior probability shift assumption).
This module is created with two purposes in mind:
  - easily apply state-of-the-art quantification algorithms to the real problems,
  - benchmark novel quantification algorithms against others.

It is compatible with any classifier using any machine learning framework.

## Contribution
Thank you for your effort!

### Reporting a bug

If you find a bug, please [submit a new issue](https://github.com/labelshift/labelshift/issues).

To be able to reproduce a bug, we will usually need the following information:

  - Versions of Python packages used (in particular version of this library).
  - A minimal code snippet allowing us to reproduce the bug.
  - What is the desired behaviour in the reported case?
  - What is the actual behaviour?


### Submitting a pull request

**Do:**

  - Do use [Google Style Guide](https://google.github.io/styleguide/pyguide.html). We use [black](https://github.com/psf/black) for code formatting.
  - Do write unit tests -- 100% code coverage is a necessity. We use [pytest](https://docs.pytest.org/).
  - Do write docstrings -- again, 100% coverage is a necessity. We use [interrogate](https://pypi.org/project/interrogate/).
  - Do consider submitting a *draft* pull request with a description of proposed changes.

**Don't:**

  - Don't include license information. This project is BSD-3 licensed and by submitting your pull request you implicitly and irrevocably agree to use this.
  - Don't implement too many ideas in a single pull request. Multiple features should be implemented in separate pull requests.

