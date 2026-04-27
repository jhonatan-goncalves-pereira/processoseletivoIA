import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# hiperparâmetros
EPOCHS      = 5
BATCH_SIZE  = 128
VAL_SPLIT   = 0.1
NUM_CLASSES = 10

SEP = "=" * 60

print("\n" + SEP)
print("  CNN MNIST — TREINAMENTO PARA EDGE AI")
print("  TensorFlow", tf.__version__)
print(SEP)


# carregamento e processamento
print("\n[1/5] Carregando dataset MNIST...")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# adiciona canal grayscale e normaliza para [0,1]
# shape: (N, 28, 28) -> (N, 28, 28, 1)
x_train = x_train[..., tf.newaxis].astype("float32") / 255.0
x_test  = x_test [..., tf.newaxis].astype("float32") / 255.0

print(f"   Treino : {x_train.shape}  dtype={x_train.dtype}")
print(f"   Teste  : {x_test.shape}   classes={NUM_CLASSES}")
print(f"   Range  : [{x_train.min():.1f}, {x_train.max():.1f}]  (normalizado)")



# 2  arquitetura CNN
print("\n[2/5] Construindo arquitetura CNN...")

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1), name="input_layer"),

        # bloco 1 extracao de bordas e texturas simples
        # padding=same mantem dimensoes 28x28 apos convolucao
        layers.Conv2D(32, kernel_size=3, activation="relu",
                      padding="same", name="conv2d_bloco1"),
        layers.MaxPooling2D(pool_size=2, name="maxpool_bloco1"),
        # saida: (14, 14, 32)

        # bloco 2: combinacao de features em padroes complexos
        # 64 filtros capturam combinacoes dos 32 mapas anteriores
        layers.Conv2D(64, kernel_size=3, activation="relu",
                      padding="same", name="conv2d_bloco2"),
        layers.MaxPooling2D(pool_size=2, name="maxpool_bloco2"),
        # saida: (7, 7, 64)

        # colapsa (7,7,64) -> (64,): 1 valor medio por mapa de feature
        # elimina 97% dos parametros vs Flatten sem perda relevante
        layers.GlobalAveragePooling2D(name="global_avg_pool"),

        # classificador compacto
        layers.Dense(64, activation="relu", name="dense_classificador"),
        layers.Dropout(0.25, name="dropout_regularizacao"),

        # saida probabilidade por classe (0-9)
        layers.Dense(NUM_CLASSES, activation="softmax", name="output_softmax"),
    ],
    name="mnist_edge_cnn",
)

model.summary()

total_params = model.count_params()
size_kb = total_params * 4 / 1024
print(f"\n   Parametros totais : {total_params:,}")
print(f"   Tamanho estimado  : {size_kb:.1f} KB (float32)")
print(f"   -> Modelo adequado para MCU/ESP32 com >=256 KB RAM")



# 3 compilacao
print("\n[3/5] Compilando modelo...")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
    ],
)

print("   Otimizador : Adam (lr=0.001)")
print("   Loss       : sparse_categorical_crossentropy")
print("   Metricas   : accuracy, top-2 accuracy")

# 4 treinamento
print("\n[4/5] Iniciando treinamento...")
print(f"   Dataset  : {len(x_train):,} treino | {len(x_test):,} teste")
print(f"   Epocas   : {EPOCHS} | Batch: {BATCH_SIZE} | Val split: {int(VAL_SPLIT*100)}%\n")

history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1,
)

# 5 avaliacao de metricas
print("\n[5/5] Avaliacao final no conjunto de teste (10.000 amostras)...")

results = model.evaluate(x_test, y_test, verbose=0)
test_loss     = results[0]
test_acc      = results[1]
test_top2_acc = results[2]

# prediz completas para avaliar
y_pred_probs = model.predict(x_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

# media por resultado
correct_mask = (y_pred == y_test)
conf_correct = float(y_pred_probs[correct_mask].max(axis=1).mean())
conf_wrong   = float(y_pred_probs[~correct_mask].max(axis=1).mean()) \
               if (~correct_mask).sum() > 0 else 0.0

# acuracia por digito
acc_per_digit = {}
for d in range(NUM_CLASSES):
    mask = (y_test == d)
    acc_per_digit[d] = (y_pred[mask] == d).sum() / mask.sum() * 100

# relatorio
print("\n" + SEP)
print("  RESULTADO FINAL — CONJUNTO DE TESTE")
print(SEP)
print(f"\n  Accuracy          : {test_acc*100:6.2f}%")
print(f"  Top-2 Accuracy    : {test_top2_acc*100:6.2f}%")
print(f"  Loss              : {test_loss:.4f}")
print(f"  Acertos absolutos : {int(test_acc*len(x_test)):,} de {len(x_test):,}")

print(f"\n  Confianca media (acertos) : {conf_correct*100:.2f}%")
print(f"  Confianca media (erros)   : {conf_wrong*100:.2f}%")
print(f"  Gap de confianca          : {(conf_correct-conf_wrong)*100:.2f}pp")
print(f"  -> Gap alto = modelo bem calibrado (nao 'chuta' a classe)")

print("\n  Acuracia por digito:")
for d in range(NUM_CLASSES):
    bar   = "█" * int(acc_per_digit[d] // 5)
    badge = "OK" if acc_per_digit[d] >= 90 else "??"
    print(f"   Digito {d}: {acc_per_digit[d]:5.2f}%  {bar} [{badge}]")

print("\n  Evolucao da validacao por epoca:")
for i, (va, vl) in enumerate(
    zip(history.history["val_accuracy"], history.history["val_loss"])
):
    bar = "#" * int(va * 20)
    print(f"   Epoca {i+1}: acc={va*100:5.2f}%  loss={vl:.4f}  {bar}")

print("\n" + SEP)
print("  INTERPRETACAO DAS METRICAS")
print(SEP)
print(f"""
  Accuracy {test_acc*100:.2f}%
  Para uma CNN com {total_params:,} parametros treinada em {EPOCHS} epocas
  em CPU, este resultado confirma arquitetura bem calibrada.
  Modelos maiores atingem 99%+, mas sao inlviaveis para MCUs.

  Top-2 Accuracy {test_top2_acc*100:.2f}%
  Em {test_top2_acc*100:.2f}% dos casos, a classe correta esta entre as 2
  maiores probabilidades. Util para sistemas com fallback
  embarcado ou threshold de confianca configuravel.

  Confianca {conf_correct*100:.2f}% (acertos) vs {conf_wrong*100:.2f}% (erros)
  Gap de {(conf_correct-conf_wrong)*100:.2f}pp indica discriminacao confiavel
  entre certeza e incerteza — propriedade critica em
  sistemas embarcados sem mecanismo de retry.

  Trade-off Edge AI: {size_kb:.0f} KB vs acuracia maxima
  Priorizamos modelo que execute em 256 KB RAM.
  Modelos maiores (~1M params) atingiriam ~99% de accuracy,
  mas sao inlviaveis para MCU/ESP32 com memoria restrita.
""")

# salvando
MODEL_PATH = "model.h5"
model.save(MODEL_PATH)
file_size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"  Modelo salvo: {MODEL_PATH}  ({file_size_kb:.1f} KB — float32 completo)")
print("\n" + SEP)
print("  TREINAMENTO CONCLUIDO COM SUCESSO")
print(SEP + "\n")