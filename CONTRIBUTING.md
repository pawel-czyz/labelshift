# Contribution guide
Thank you for your time!

## Reporting a bug

If you find a bug, please [submit a new issue](https://github.com/labelshift/labelshift/issues).

To be able to reproduce a bug, we will usually need the following information:

  - Versions of Python packages used (in particular version of this library).
  - A minimal code snippet allowing us to reproduce the bug.
  - What is the desired behaviour in the reported case?
  - What is the actual behaviour?


## Submitting a pull request

**Do:**

  - Do use [Google Style Guide](https://google.github.io/styleguide/pyguide.html). We use [black](https://github.com/psf/black) for code formatting.
  - Do write unit tests – 100% code coverage is a necessity. We use [pytest](https://docs.pytest.org/).
  - Do write docstrings – 100% coverage is a necessity. We use [interrogate](https://pypi.org/project/interrogate/).
  - Do write high-level documentation as examples and tutorials, illustrating introduced features.
  - Do consider submitting a *draft* pull request with a description of proposed changes.
  - Do check the [Development section](#development).

**Don't:**

  - Don't include license information. This project is BSD-3 licensed and by submitting your pull request you implicitly and irrevocably agree to use this.
  - Don't implement too many ideas in a single pull request. Multiple features should be implemented in separate pull requests.

## Development
To install the repository in editable mode use:
```
pip install -r requirements.txt  # Install dev requirements
pip install -e .  # Install the module in editable mode
pre-commit install  # Install pre-commit hooks
```
We suggest using a virtual environment for this.

You can use `make` to run the required code quality checks.


Thank you a lot!
