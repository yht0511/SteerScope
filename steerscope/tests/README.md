## How to run our script-based tests for released datasets

These optional checks validate separately downloaded Concept10, Concept500 and
Concept16K artifacts. Missing releases are skipped during the normal test suite.

```bash
STEERSCOPE_RELEASE_ROOT=/path/to/output \
STEERSCOPE_MODEL_ROOT=/path/to/results \
python -m pytest steerscope/tests/test_released_artifacts.py
```


## How to run our unit tests

Once the dataset test passes, you can run the unit tests for functonal modules.

```bash
python -m pytest steerscope/tests/unit_tests
```
