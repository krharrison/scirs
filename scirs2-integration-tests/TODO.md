# scirs2-integration-tests TODO

## Status: v0.3.2

## Purpose

Cross-crate integration tests for SciRS2 ecosystem.

## v0.3.2 Coverage

- autograd + neural integration
- linalg + sparse interop
- stats + optimize integration
- signal + fft pipeline
- vision + ndimage pipeline

## v0.4.0 Planned Tests

- [ ] End-to-end ML pipeline (datasets -> neural -> optimize -> metrics)
- [ ] Full signal analysis pipeline (io -> signal -> stats -> series)
- [ ] Computer vision pipeline (io -> ndimage -> vision -> metrics)
- [ ] Graph ML pipeline (graph -> neural -> metrics)
- [ ] Scientific computing pipeline (integrate -> linalg -> sparse)
- [ ] NLP pipeline (text -> neural -> metrics)

## Running Tests

cargo nextest run --all-features -p scirs2-integration-tests
