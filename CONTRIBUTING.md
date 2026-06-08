## How to contribute

### Reporting bugs
If you find a bug in Basix, please report it on the [GitHub issue tracker](https://github.com/fenics/basix/issues/new?labels=bug).

### Suggesting enhancements
If you want to suggest a new feature or an improvement of a current feature, you can submit this
on the [issue tracker](https://github.com/fenics/basix/issues).

### Submitting a pull request
If you want to directly submit code to Basix, you can do this by forking the repo, then submitting a pull request.
If you want to contribute, but are unsure where to start, have a look at the
[issues labelled "good first issue"](https://github.com/FEniCS/basix/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

On opening a pull request, unit tests will run on GitHub Actions. You can click on these in the pull request
to see where (if anywhere) the tests are failing.

The GitHub Actions runs include `ruff` formatting and checking. Before opening a PR, you can
run `ruff format` locally to reformat your code and `ruff check` to locally check for code
suggestion that the CI run will pick up on.

### Code of conduct
We expect all our contributors to follow the [code of conduct](CODE_OF_CONDUCT.md). Any unacceptable
behaviour can be reported to the FEniCS steering council (fenics-steering-council@googlegroups.com).
