# Contributing to CAMPUSCARE

CAMPUSCARE is a Django-based campus complaint triage prototype for an online coding event. Open GitHub Issues define the tasks and carry an `easy`, `medium`, or `hard` difficulty label.

## Getting started

1. Read the open Issues and choose one that matches your experience.
2. Comment on the issue before starting so contributors can coordinate.
3. Fork or clone the repository and create a branch.
4. Create and activate a virtual environment.
5. Install dependencies from `requirements.txt`.
6. Run migrations and start the Django development server.

## Development workflow

Keep changes focused on the selected issue and preserve the existing complaint submission flow. Use clear Python and Django code that another student can understand.

For ML changes, describe the examples or dataset used, report before/after results, and test category prediction and priority assignment separately. Small hand-written examples are useful for regression tests but do not prove real-world model accuracy. Document limitations, especially where training data is synthetic or limited.

## Before submitting

Run:

```bash
python manage.py check
python manage.py test
```

Then verify the complaint submission flow manually. Open a Pull Request that explains the change, how it was tested, and links the related issue (for example, `Closes #2`).

## Do not commit

- Virtual environments or generated cache files
- Passwords, secret keys, or environment files containing secrets
- Real student personal or sensitive information

Difficulty labels describe expected scope, not guaranteed completion time.