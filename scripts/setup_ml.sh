#!/bin/bash

# 이 스크립트는 FLIGHTSTACK_SIM (프로젝트 루트)에서 실행한다고 가정합니다.

echo "=== 머신러닝(SVM) 가상환경 자동 세팅을 시작합니다 ==="

# 1. venv 폴더가 없으면 새로 생성
if [ ! -d "venv" ]; then
    echo ">> 가상환경(venv) 폴더를 생성합니다..."
    python3 -m venv venv
else
    echo ">> 가상환경(venv)이 이미 존재합니다. 생성을 생략합니다."
fi

# 2. 가상환경 활성화
echo ">> 가상환경을 활성화합니다..."
source venv/bin/activate

# 3. 패키지 설치 (pip 최신화 후 설치)
echo ">> 필수 패키지를 설치합니다..."
pip install --upgrade pip
pip install scikit-learn pandas numpy matplotlib pyulog pymavlink torch torchvision torchaudio pybinlog

echo "=== 세팅이 모두 완료되었습니다! ==="