import time
import math

import utils
import prime_factorization as pf

def test(factorize_f):
    for i in range(2, 24):
        n = (2 ** i) - 1
        print(f"{n}:{factorize_f(n)}")

def test_prime_factorize_elapsed_time(n):
    start = time.time()
    print(f"{n}:{pf.prime_factorize(n)}")
    end = time.time()
    print(f"prime_factorize elapsed time: {end - start}")

def test_prime_factorize_by_primes_table_elapsed_time(n, primes):
    start = time.time()
    print(f"{n}:{pf.prime_factorize_by_primes_table(n, primes)}")
    end = time.time()
    print(f"prime_factorize_by_primes_table elapsed time: {end - start}")

def test_prime_factorize_by_mpf_elapsed_time(n):
    start = time.time()
    print(f"{n}:{pf.prime_factorize_by_mpf(n)}")
    end = time.time()
    print(f"prime_factorize_by_mpf elapsed time: {end - start}")

def test_find_primes_elapsed_time(find_primes_f, n):
    start = time.time()
    find_primes_f(n)
    end = time.time()
    print(f"{n}: find_primes_elapsed_time elapsed time: {end - start}")


# test(pf.prime_factorize)

# n = (2 ** 42) - 1
# sqrtn = math.isqrt(n)
# primes = utils.find_primes_by_sieve(sqrtn)

# test_prime_factorize_elapsed_time(n)
# test_prime_factorize_by_primes_table_elapsed_time(n, primes)

# test_find_primes_elapsed_time(utils.find_primes, sqrtn)
# test_find_primes_elapsed_time(utils.find_primes_by_sieve, sqrtn)

n = (2 ** 128) - 1
test_prime_factorize_by_mpf_elapsed_time(n)