# Exercises: Testing + Debugging

Create a branch: `git switch -c exercise/07-testing`

## A. Write a function
Create `is_prime.py` with function `is_prime(n)`.

## B. Add tests
Create `test_is_prime.py`:
- test small primes (2,3,5,7)
- test non-primes (1,4,9,21)
- test negatives return False

Run:
```bash
pytest -q
```

## Git checkpoint
Commit function, then commit tests.
