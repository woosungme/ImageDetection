import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 잘린 이미지에 대한 에러 핸들링
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 데이터셋 경로 설정
train_dir = '/content/drive/MyDrive/Colab Notebooks/image_data_ws/train'
test_dir = '/content/drive/MyDrive/Colab Notebooks/image_data_ws/test'

# 이미지 데이터 전처리
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 이미지 스케일링
    rotation_range=40,  # 무작위 회전 범위
    width_shift_range=0.2,  # 수평 이동 범위
    height_shift_range=0.2,  # 수직 이동 범위
    shear_range=0.2,  # 전단 변환 범위
    zoom_range=0.2,  # 확대 축소 범위
    horizontal_flip=True,  # 수평 뒤집기
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# 데이터셋 로드
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'  # 이진 분류
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # 테스트 데이터는 순서를 유지
)

# 데이터셋 샘플 수 출력
print(f"Total training samples: {train_generator.samples}")
print(f"Total testing samples: {test_generator.samples}")

# 모델 생성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # 한 에포크당 100번의 배치를 사용하여 훈련
    epochs=15,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size  # 검증 단계 수
)

# 모델 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")
