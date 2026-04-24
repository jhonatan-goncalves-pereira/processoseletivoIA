import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#carregamento e pré-processamento do MNIST

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Adiciona dimensão de canal (grayscale) e normaliza para [0, 1]
# shape: (N, 28, 28) -> (N, 28, 28, 1)
x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
x_test  = x_test [..., tf.newaxis].astype("float32") / 255.0

# ──────────────────────────────────────────────
# arquitetura CNN para Edge AI
#
# decisão de design:
#   - apenas 2 blocos Conv+Pool: suficiente para MNIST (imagens 28×28,
#     10 classes simples), evita overfitting e mantém o modelo leve.
#   - filtros pequenos (16 e 32): reduz parâmetros sem perda relevante
#     de acurácia, ideal para inferência em CPU/MCU.
#   - globalAveragePooling2D no lugar de Flatten: elimina parâmetros
#     desnecessários e atua como regularizador implícito.
#   - camada Dense única de 64 neurônios: trade-off entre capacidade
#     e tamanho do modelo. Mais neurônios = mais acurácia, mas mais
#     peso — para Edge AI, 64 é o ponto ideal no MNIST.
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),

        # bloco 1  extrai bordas e texturas simples
        layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=2),

        # bloco 2 combina características em padrões mais complexos
        layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=2),

        # ceduz mapa de características para 1 valor por filtro
        layers.GlobalAveragePooling2D(),

      
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),          
        layers.Dense(10, activation="softmax"),
    ],
    name="mnist_edge_cnn",
)

model.summary()

#  compilação e treinamento
# metricas monitoradas:
#   - accuracy  métrica principal de classificação
#   - AUC mede discriminação entre classes (area sob a curva ROC)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\n Iniciando treinamento...\n")

history = model.fit(
    x_train, y_train,
    epochs=5,                  
    batch_size=128,
    validation_split=0.1,      
    verbose=1,
)


print("\n Avaliação no conjunto de teste:")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f"  → Loss     : {test_loss:.4f}")
print(f"  → Accuracy : {test_acc * 100:.2f}%")

#salvamento do modelo
model.save("model.h5")
print("\nModelo salvo em: model.h5")