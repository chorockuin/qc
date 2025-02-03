import math  # 수학 관련 함수들을 제공하는 모듈
import random
import numpy as np
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

import utils  # utils 모듈은 add_factor, get_mpf 등 유틸리티 함수를 제공한다고 가정

# 기본 소인수 분해 함수
def prime_factorize(n):
    factors = {}            # 소인수와 그 지수(개수)를 저장할 딕셔너리
    sqrtn = math.isqrt(n)   # n의 정수 제곱근을 계산하여, 반복 범위를 √n으로 제한
    
    # 2부터 √n까지 모든 숫자에 대해 소인수 여부를 검사
    for i in range(2, sqrtn + 1):
        # i가 n의 약수인 동안 반복 (여러 번 나누어질 수 있으므로 while문 사용)
        while (n % i == 0):
            utils.add_factor(factors, i)  # utils 모듈의 add_factor 함수를 사용하여 factors에 i를 추가
            n = n // i                    # n을 i로 나누어 소인수 분해 과정을 진행
    
    # 반복이 끝난 후, n이 1보다 크면 남은 n은 소수이므로 결과에 추가
    if (n > 1):
        utils.add_factor(factors, n)
    
    return factors  # 소인수와 그 개수가 담긴 딕셔너리 반환


# 미리 생성된 소수(primes) 리스트를 활용한 소인수 분해 함수
def prime_factorize_by_primes_table(n, primes):
    factors = {}  # 소인수와 그 지수를 저장할 딕셔너리
    
    # primes 리스트에 포함된 소수들에 대해 n을 나누어 소인수를 추출
    for i in primes:
        # i가 n의 약수인 동안 계속해서 나누어준다.
        while (n % i == 0):
            utils.add_factor(factors, i)  # i를 소인수 딕셔너리에 추가
            n = n // i                    # n을 i로 나누어 소인수 분해 진행
    
    # 모든 primes로 나눈 후 남은 n이 1보다 크면, 남은 n 자체가 소인수(소수가 아닐 수도 있음)로 처리
    if (n > 1):
        utils.add_factor(factors, n)
    
    return factors  # 소인수 분해 결과 반환


# 최소 소인수(MPF; minimal prime factor)를 이용한 소인수 분해 함수
def prime_factorize_by_mpfs(n):
    mpfs = {}     # 최소 소인수를 캐싱하기 위한 딕셔너리 (메모이제이션용)
    factors = {}  # 최종 소인수와 그 개수를 저장할 딕셔너리
    
    # n이 1이 될 때까지 최소 소인수를 구해 나눈다.
    while (n > 1):
        # utils.get_mpf 함수를 사용해 n의 최소 소인수를 구한다.
        # mpfs 캐시를 활용하여 이미 계산한 결과를 재사용할 수 있음.
        mpfn = utils.get_mpf(n, mpfs)
        utils.add_factor(factors, mpfn)  # 최소 소인수를 결과 딕셔너리에 추가
        n = n // mpfn                 # n을 최소 소인수로 나누어 분해 과정을 진행
    
    return factors  # 소인수 분해 결과 반환


# 최소 소인수 테이블(mpfs 테이블)을 활용한 소인수 분해 함수 (아직 구현되지 않음)
def prime_factorize_by_mpfs_table(n, mpfs):
    # TODO: 최소 소인수 테이블(mpfs)을 활용하여 n의 소인수 분해를 구현하세요.
    pass

def find_period(n, a):
    for x in range(1, n):
        if (a ** x) % n == 1:
            return x
    return -1

def mod_pow(a, x, n):
    y = 1
    while x > 0:
        if (x & 1) == 1:
            y = (y * a) % n
        x = x >> 1
        a = (a * a) % n
    return y

def find_period_by_mod_pow(n, a):
    for x in range(1, n):
        if mod_pow(a, x, n) == 1:
            return x
    return -1

def prime_factorize_by_shor(n):
    qc = None
    while (True):
        a = random.randint(2, n-1)
        gcd = math.gcd(n, a)
        if (gcd != 1):
            return gcd, n // gcd, qc
        # r = find_period(n, a)
        r = find_period_by_mod_pow(n, a)
        if (r % 2 != 0):
            continue
        gcd1 = math.gcd(n, a ** (r//2) + 1)
        gcd2 = math.gcd(n, a ** (r//2) - 1)
        if gcd1 == 1 or gcd2 == 1:
            continue
        return gcd1, gcd2, qc

def find_period_by_quantum_circuit(n, a):
    phase, qc = qpe_amod15(a)
    frac = Fraction(phase).limit_denominator(15)
    return frac.denominator, qc

def qpe_amod15(a):
    n_count = 3
    total_qubits = 4 + n_count  # work qubits 4개와 counting qubits n_count개
    qc = QuantumCircuit(total_qubits, n_count)
    
    # 1. Counting qubits에 Hadamard 게이트 적용
    for q in range(n_count):
        qc.h(q)
    
    # 2. work qubits 중 마지막 큐비트(인덱스 3+n_count)에 X 게이트 적용
    qc.x(3 + n_count)
    
    # 3. 각 counting qubit에 대해 제어 연산 (c_amod15 게이트) 적용
    for q in range(n_count):
        # 제어 대상: counting qubit q와 work qubits [n_count, n_count+1, n_count+2, n_count+3]
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])
    
    # 4. Counting qubits에 역 QFT (qft_dagger) 적용
    qc.append(qft_dagger(n_count), list(range(n_count)))
    
    # 5. Counting qubits 측정
    qc.measure(list(range(n_count)), list(range(n_count)))
    
    # 6. AerSimulator 백엔드 사용
    backend = AerSimulator()
    
    # 7. 회로를 분해하여 기본 게이트로 변환 (transpile 사용)
    qc_decomposed = transpile(qc, backend=backend)
    
    # 8. 실행
    job = backend.run(qc_decomposed, shots=1, memory=True)
    result = job.result()
    
    # 9. 측정 결과 처리
    readings = result.get_memory()
    phase = int(readings[0], 2) / (2 ** n_count)
    
    return phase, qc


def c_amod15(a, power):
    if a not in [2, 7, 8, 11, 13]:
        raise ValueError("'a' must be in [2, 7, 8, 11, 13]")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2, 13]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a == 11:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = " %i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = " QFT† (I-QFT)"
    return qc

def prime_factorize_by_qc(n):
    trial = 0
    while (True):
        trial += 1
        print('trial =', trial)
        a = random.randint(2, n - 1)
        if a not in [2, 7, 8, 11, 13]:
            continue
        r, qc = find_period_by_quantum_circuit(n, a)
        print('\ta =', a, 'r =', r)
        if (r % 2 != 0):
            continue
        gcd1 = math.gcd(n, a ** (r // 2) + 1)
        gcd2 = math.gcd(n, a ** (r // 2) - 1)
        print('\tgcd1 =', gcd1, 'gcd2 =', gcd2)
        if (gcd1 == 1 or gcd2 == 1):
            continue
        return gcd1, gcd2, qc