import math

def add_factor(factors, f):
    # 소인수 분해 결과를 저장하는 딕셔너리 factors에
    # 소인수 f를 추가합니다.
    # 만약 f가 이미 존재하면 해당 소인수의 개수를 1 증가시킵니다.
    if f in factors:
        factors[f] += 1
    else:
        factors[f] = 1

def find_primes(n):
    # 2부터 n까지의 모든 소수를 찾아 리스트로 반환하는 함수입니다.
    primes = []
    # 2부터 n까지 순회하며 각 숫자가 소수인지 검사합니다.
    for i in range(2, n + 1):
        if is_prime(i):  # i가 소수라면
            primes.append(i)  # 소수 리스트에 추가
    return primes

def is_prime(n):
    # 주어진 n이 소수인지 여부를 판별하는 함수입니다.
    # n의 제곱근까지만 검사하면 충분합니다.
    sqrtn = math.isqrt(n)  # n의 정수 제곱근을 구합니다.
    # 2부터 n의 제곱근까지의 모든 수로 n을 나누어 봅니다.
    for i in range(2, sqrtn + 1):
        if (n % i == 0):  # i가 n의 약수라면
            return False  # n은 소수가 아님
    return True  # 약수가 없으면 n은 소수임

def find_primes_by_sieve(n):
    # 0과 1은 소수가 아니므로 0으로, 2부터 n까지는 초기에는 모두 소수라고 가정하여 1로 설정.
    # flags 리스트의 인덱스는 해당 숫자를 의미하며, 값이 1이면 소수(아직 후보), 0이면 소수가 아님을 의미.
    flags = [0, 0] + [1] * (n - 1)
    
    # n의 정수 제곱근을 계산. 소수 판별은 2부터 sqrt(n)까지 진행하면 충분하다.
    sqrtn = math.isqrt(n)
    
    # 2부터 sqrt(n)까지 반복하며 소수(또는 소수 후보)를 찾는다.
    for i in range(2, sqrtn + 1):
        # 만약 i가 아직 소수 후보라면 (flags[i]가 1이면)
        if flags[i] == 1:
            # i의 제곱(i * i)부터 n까지, i씩 증가하면서 i의 배수를 모두 소수가 아니라고 표시.
            # i보다 작은 배수들은 이미 다른 소수에 의해 제거되었으므로, i*i부터 시작하는 것이 효율적이다.
            for j in range(i * i, n + 1, i):
                flags[j] = 0  # j는 i의 배수이므로 소수가 아니다.
    
    # 최종적으로 flags 배열에서 값이 1인 인덱스들이 소수임.
    primes = []
    for i in range(len(flags)):
        if flags[i] == 1:
            primes.append(i)
    
    # n 이하의 모든 소수 리스트를 반환.
    return primes

def get_mpf(n, mpfs):
    # mpfs: {숫자: 해당 숫자의 최소 소인수}를 저장하는 캐시(딕셔너리)
    # 만약 n의 최소 소인수가 이미 계산되어 있다면, 바로 반환한다.
    if n in mpfs:
        return mpfs[n]
    else:
        # n의 정수 제곱근을 계산 (소인수 검사를 위한 범위 한계)
        sqrtn = math.isqrt(n)
        # 2부터 n의 제곱근까지 반복하며 n의 약수를 찾는다.
        for i in range(2, sqrtn + 1):
            # 만약 i가 n의 약수라면, i는 n의 최소 소인수이다.
            if (n % i) == 0:
                # n의 최소 소인수를 캐시에 저장
                mpfs[n] = i
                return i
        # 반복문을 다 돌았는데 약수를 찾지 못했다면, n은 소수이므로 n 자체가 최소 소인수이다.
        mpfs[n] = n
        return n
