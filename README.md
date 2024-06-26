# ImageDetection
normal-abnormal image detection Using CNN

train data : 499개
test data : 125개

![스크린샷 2024-05-08 오후 2 51 03](https://github.com/woosungme/ImageDetection/assets/163384279/cd8b6726-a8d3-44e6-a703-d6af305d01f9)
![스크린샷 2024-05-08 오후 2 51 19](https://github.com/woosungme/ImageDetection/assets/163384279/14eecd45-a2cc-456c-83bb-989bf78fe3c2)

### 모델 성능 평가

#### 훈련 결과

- **손실 (Loss)**: 최종 에포크에서 0.2517
- **정확도 (Accuracy)**: 최종 에포크에서 0.9186

#### 검증 결과

- **검증 손실 (Validation Loss)**: 최종 에포크에서 0.2719
- **검증 정확도 (Validation Accuracy)**: 최종 에포크에서 0.9062

#### 테스트 결과

- **테스트 손실 (Test Loss)**: 0.2246
- **테스트 정확도 (Test Accuracy)**: 0.9280

### 결과 해석

훈련 과정에서는 초기에 손실과 정확도가 변동이 큰 경향을 보였으나, 에포크가 진행됨에 따라 손실이 감소하고 정확도가 상승하는 경향을 보였습니다. 검증 데이터에 대해서도 유사한 경향을 보였으며, 테스트 데이터에서도 높은 정확도를 얻었습니다.

### 결론

최종적으로 모델은 훈련, 검증, 테스트 데이터 모두에서 높은 성능을 보이고 있으며, 과적합 현상도 크게 나타나지 않고 있습니다. 이러한 결과는 모델이 주어진 데이터셋에 대해 효과적으로 학습되었고, 일반화 성능도 뛰어남을 시사합니다.
긍정적인 방향으로 확인하였고, 더 많은 데이터로 학습을 진행해 볼 예정입니다.

