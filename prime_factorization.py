import math
import utils

import math  # 수학 관련 함수들을 제공하는 모듈
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
