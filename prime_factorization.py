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
    # 주어진 'a'에 대해 모듈로 15 연산의 주기를 찾기 위해 양자 회로를 실행합니다.
    phase, qc = qpe_amod15(a)
    # 측정된 위상(phase)를 Fraction을 사용하여 유리수로 변환하고,
    # 분모를 최대 15까지 근사합니다. (분모가 주기가 됩니다.)
    frac = Fraction(phase).limit_denominator(15)
    # 주기(분모)와 사용된 양자 회로를 반환합니다.
    return frac.denominator, qc

def qpe_amod15(a):
    # QPE(Quantum Phase Estimation)를 통한 모듈로 15 연산의 위상 추정을 구현합니다.
    
    # 계수(qubit) 수를 3으로 설정 (위상 추정에 사용될 비트 수)
    n_count = 3
    # 전체 사용될 qubit 수는 작업 레지스터 4개와 계수 레지스터 n_count개입니다.
    total_qubits = 4 + n_count
    # 총 total_qubits 개의 큐비트와 n_count 개의 클래식 비트를 가지는 양자 회로 생성
    qc = QuantumCircuit(total_qubits, n_count)
    
    # 계수 레지스터(0 ~ n_count-1번 큐비트)에 Hadamard 게이트 적용하여 균등한 중첩 상태 생성
    for q in range(n_count):
        qc.h(q)
    
    # 작업 레지스터 초기화를 위해 n_count 오프셋 이후의 4번째 큐비트에 X 게이트 적용
    qc.x(3 + n_count)
    
    # 각 계수 큐비트에 대해 제어된 모듈로 15 연산(모듈러 거듭제곱)을 적용
    for q in range(n_count):
        # c_amod15 함수로부터 a^(2^q) mod 15 연산의 제어 게이트를 받아 적용
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])
    
    # 계수 레지스터에 대해 역 QFT (Quantum Fourier Transform†) 적용
    qc.append(qft_dagger(n_count), list(range(n_count)))
    
    # 계수 큐비트들을 고전 비트에 측정하여 위상 값을 추출
    qc.measure(list(range(n_count)), list(range(n_count)))
    
    # 시뮬레이터 백엔드(AerSimulator)를 사용하여 회로 실행
    backend = AerSimulator()
    
    # 회로를 시뮬레이터에 맞게 트랜스파일(최적화)합니다.
    qc_decomposed = transpile(qc, backend=backend)
    
    # 시뮬레이터에서 회로를 1회 실행하며 각 샷의 메모리를 저장
    job = backend.run(qc_decomposed, shots=1, memory=True)
    result = job.result()
    
    # 측정 결과(비트 문자열)를 메모리에서 추출
    readings = result.get_memory()
    # 측정된 비트 문자열을 정수로 변환한 후, 전체 상태 개수(2^n_count)로 나누어 위상 값을 계산
    phase = int(readings[0], 2) / (2 ** n_count)
    
    # 계산된 위상과 해당 양자 회로를 반환
    return phase, qc

def c_amod15(a, power):
    # 'a' 값이 허용된 값(2, 7, 8, 11, 13)인지 확인
    if a not in [2, 7, 8, 11, 13]:
        raise ValueError("'a' must be in [2, 7, 8, 11, 13]")
    
    # 4 큐비트를 사용하는 회로 U를 생성 (모듈러 곱셈을 구현할 작업 레지스터)
    U = QuantumCircuit(4)
    
    # 지정된 power 만큼 반복하며 a^(power) mod 15 연산을 구현
    for iteration in range(power):
        if a in [2, 13]:
            # a가 2 또는 13인 경우: 큐비트 간 swap 연산을 통해 곱셈 연산 수행
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        if a in [7, 8]:
            # a가 7 또는 8인 경우: 다른 순서의 swap 연산 적용
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a == 11:
            # a가 11인 경우: 특수한 swap 연산 적용
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            # a가 7, 11, 13인 경우: 모든 큐비트에 X 게이트 적용 (비트 플립)
            for q in range(4):
                U.x(q)
    
    # 생성된 회로 U를 하나의 게이트로 변환
    U = U.to_gate()
    # 게이트에 이름 지정 (예: " 7^2 mod 15")
    U.name = " %i^%i mod 15" % (a, power)
    # U의 제어 게이트(Controlled-U)를 생성하여 반환 (양자 위상 추정에 사용)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    # n 큐비트에 대한 역 양자 푸리에 변환(QFT†) 회로를 생성합니다.
    qc = QuantumCircuit(n)
    
    # 큐비트 순서를 반전시키기 위해 스왑 게이트를 적용 (대칭화)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    
    # 역 QFT의 핵심: 제어 위상 회전 게이트와 Hadamard 게이트를 적절한 순서로 적용
    for j in range(n):
        for m in range(j):
            # m번째 큐비트에서 j번째 큐비트로 -π/2^(j-m) 회전하는 제어 위상 게이트(cp) 적용
            qc.cp(-np.pi / float(2**(j-m)), m, j)
        # 각 큐비트에 Hadamard 게이트 적용
        qc.h(j)
    
    # 회로에 이름 지정 (역 QFT임을 명시)
    qc.name = " QFT† (I-QFT)"
    return qc

def prime_factorize_by_qc(n):
    # 양자 회로를 이용하여 정수 n의 소인수분해(Shor의 알고리즘)를 수행합니다.
    trial = 0
    while (True):
        trial += 1
        print('trial =', trial)
        # 2와 n-1 사이에서 무작위로 정수 a 선택
        a = random.randint(2, n - 1)
        # a가 허용된 값이 아니면 건너뜁니다.
        if a not in [2, 7, 8, 11, 13]:
            continue
        # 선택한 a에 대해 주기 r을 찾고, 관련 양자 회로를 생성
        r, qc = find_period_by_quantum_circuit(n, a)
        print('\ta =', a, 'r =', r)
        # 주기 r이 짝수가 아니라면 유효한 인수분해로 이어지지 않으므로 건너뜁니다.
        if (r % 2 != 0):
            continue
        # a^(r/2) ± 1과 n의 최대공약수를 계산하여 소인수를 찾습니다.
        gcd1 = math.gcd(n, a ** (r // 2) + 1)
        gcd2 = math.gcd(n, a ** (r // 2) - 1)
        print('\tgcd1 =', gcd1, 'gcd2 =', gcd2)
        # 비자명한 인수(1이 아닌)가 발견되지 않으면 다시 시도합니다.
        if (gcd1 == 1 or gcd2 == 1):
            continue
        # 비자명한 두 인수와 사용된 양자 회로를 반환합니다.
        return gcd1, gcd2, qc
