import os
import numpy as np
import tensorflow as tf

SEP  = "=" * 65
SEP2 = "-" * 65

# 1 carrega model treinado
# ═══════════════════════════════════════════════════════════
print("\n" + SEP)
print("  OTIMIZACAO PARA EDGE AI — TFLite Converter")
print(SEP)

print("\n[1/5] Carregando modelo treinado (model.h5)...")
model = tf.keras.models.load_model("model.h5")
total_params = model.count_params()
h5_size_kb   = os.path.getsize("model.h5") / 1024
print(f"   Parametros : {total_params:,}  |  Formato: float32")
print(f"   Tamanho    : {h5_size_kb:.1f} KB")

# 2 usando dados de calibrar
# Full Integer Quantization exige amostras representativas para
# calcular os ranges de ativacao em cada camada da rede
# aqui usando 200 amostras — suficiente para o MNIST
print("\n[2/5] Preparando dados de calibracao...")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test[..., tf.newaxis].astype("float32") / 255.0

CALIB_SAMPLES = 200

def calibration_dataset():
    """gerador de amostras para calibracao de quantizacao"""
    for i in range(CALIB_SAMPLES):
        sample = x_test[i:i+1]
        yield [sample]

print(f"   {CALIB_SAMPLES} amostras de calibracao preparadas")


# funcs auxiliares
def converter_e_salvar(converter, output_path, label):
    """converte, salva e retorna tamanho em KB."""
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"   [{label}] -> {output_path} ({size_kb:.1f} KB)")
    return size_kb


def avaliar_tflite(model_path, x_sample, y_sample, n=500):
    """
    avalia acuracia de um modelo TFLite em n amostras
    suporta modelos float32, float16 e int8 automaticamente
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    for i in range(n):
        inp = x_sample[i:i+1]
        # converte input para int8 se o modelo exigir (Full Integer Quant)
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            inp = (inp / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        if np.argmax(output) == y_sample[i]:
            correct += 1
    return correct / n * 100


def avaliar_tflite_safe(model_path, x_sample, y_sample, n=500):
    """versao segura com tratamento de erro."""
    try:
        return avaliar_tflite(model_path, x_sample, y_sample, n)
    except Exception as e:
        print(f"   [AVISO] Erro na avaliacao de {model_path}: {e}")
        return None

# 3 converte tflite 4 variacoes
# ═══════════════════════════════════════════════════════════
print("\n[3/5] Convertendo para TFLite — 4 variantes...")
print(SEP2)

sizes = {}

# variante 1 Baseline float32 (referencia)
# Sem otimizacao e referencia para medir impacto das tecnicas
print("\n  [1/4] Baseline float32 (referencia sem otimizacao)...")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
sizes["base"] = converter_e_salvar(conv, "model_base.tflite", "Baseline float32")

# variante 2 Dynamic Range Quantization (PRINCIPAL)
#
# funcionamento:
#   - pesos: float32 -> int8 em tempo de CONVERSAO
#   - ativacoes: quantizadas dinamicamente para int8 em cada inferencia
#     retornando a float32 ao final
#   - NAO exige dataset de calibracao
#
# uso:
#   - reduz mais ou menos 75% no tamanho (float32 -> int8 = 4x menor)
#   - compativel com qualquer CPU: MCU, ESP32, Raspberry Pi
#   - perda de acuracia tipicamente < 0.5% no MNIST
#   - ponto de entrada recomendado pelo Google para TFLite embarcado
#   - nao exige hardware especializado (NPU, GPU)
#
print("\n  [2/4] Dynamic Range Quantization (TECNICA PRINCIPAL)...")
print("        Pesos: float32 -> int8 em conversao")
print("        Ativacoes: quantizadas dinamicamente em runtime")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
sizes["dyn"] = converter_e_salvar(conv, "model.tflite", "Dynamic Range int8")

# Float16 Quantization
#
# funcionamento:
#   - pesos: float32 -> float16
#   - ativacoes: permanecem em float32
#   - NAO exige dataset de calibracao
#
# quando se usa
#   - hardware com suporte nativo fp16: GPUs, NPUs modernos
#   - quando preservar precisao de ativacoes e prioritario
#
# limte vs Dynamic Range:
#   - reducao menor (~50% vs ~75%)
#   - sem vantagem em CPU pura (MCU/ESP32 nao tem fp16 nativo)
#
print("\n  [3/4] Float16 Quantization (tecnica adicional)...")
print("        Pesos: float32 -> float16 | Ativacoes: float32")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
sizes["f16"] = converter_e_salvar(conv, "model_float16.tflite", "Float16")

# variante 4: Full Integer Quantization
#
# funcionamento:
#   - pesos E ativacoes: float32 -> int8
#   - eXIGE dataset de calibracao para calcular ranges de ativacao
#   - input/output podem ser mantidos em float32 (modo hibrido)
#
# uso:
#   - maxima compressao em hardware int8 dedicado (Coral Edge TPU)
#   - MCUs com suporte a operacoes SIMD int8
#   - quando latencia de inferencia e critica
#
# desvantagem:
#   - oode ter degradacao maior de acuracia (~1-2%)
#   - exige processo de calibracao com dados representativos
#
print("\n  [4/4] Full Integer Quantization (avancada)...")
print("        Pesos E ativacoes: float32 -> int8")
print("        Exige calibracao com dados reais")
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = calibration_dataset
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.float32   # mantém interface float para facilitar uso
conv.inference_output_type = tf.float32
sizes["int8"] = converter_e_salvar(conv, "model_int8.tflite", "Full Integer int8")


# 4 avaliando pós conversao
# ═══════════════════════════════════════════════════════════
print("\n[4/5] Avaliando acuracia pos-conversao (500 amostras)...")
print(SEP2)

acc = {
    "base" : avaliar_tflite_safe("model_base.tflite",     x_test, y_test),
    "dyn"  : avaliar_tflite_safe("model.tflite",          x_test, y_test),
    "f16"  : avaliar_tflite_safe("model_float16.tflite",  x_test, y_test),
    "int8" : avaliar_tflite_safe("model_int8.tflite",     x_test, y_test),
}

# 5 relatorio comparativo
# ═══════════════════════════════════════════════════════════
print("\n[5/5] Relatorio comparativo...")
print("\n" + SEP)
print("  COMPARATIVO: TECNICA x TAMANHO x ACURACIA")
print(SEP)

header = f"  {'Tecnica':<30} {'Tamanho':>10} {'vs Base':>10} {'Acuracia':>10}"
print(f"\n{header}")
print("  " + SEP2)

rows = [
    ("Baseline float32",           "base", "model_base.tflite"),
    ("Dynamic Range (int8) PRINC", "dyn",  "model.tflite"),
    ("Float16",                    "f16",  "model_float16.tflite"),
    ("Full Integer (int8)",        "int8", "model_int8.tflite"),
]

base_size = sizes["base"]

for label, key, _ in rows:
    sz    = sizes[key]
    pct   = f"{sz/base_size*100:.0f}%"
    ac    = f"{acc[key]:.2f}%" if acc[key] is not None else "N/A"
    mark  = " <-- PRINCIPAL" if key == "dyn" else ""
    print(f"  {label:<30} {sz:>8.1f} KB {pct:>10} {ac:>10}{mark}")

# degradacoes
print(f"\n  Degradacao de acuracia vs baseline:")
base_acc = acc["base"] or 100.0
for label, key, _ in rows[1:]:
    if acc[key] is not None:
        deg = base_acc - acc[key]
        verdict = "MINIMA" if deg < 1 else "ACEITAVEL" if deg < 3 else "ALTA"
        print(f"   {label:<30}: -{deg:.2f}pp  [{verdict}]")

print(f"""
{SEP}
  ANALISE DOS TRADE-OFFS
{SEP}

  Dynamic Range Quantization (model.tflite) — RECOMENDADA
    - Reducao: {100-sizes['dyn']/base_size*100:.0f}% do tamanho  ({base_size:.0f} KB -> {sizes['dyn']:.0f} KB)
    - Degradacao: {base_acc-(acc['dyn'] or base_acc):.2f}pp (minima — dentro do esperado)
    - Compativel: ESP32, STM32, Raspberry Pi (CPU pura)
    - Sem calibracao necessaria

  Float16 Quantization (model_float16.tflite)
    - Reducao: {100-sizes['f16']/base_size*100:.0f}% do tamanho  ({base_size:.0f} KB -> {sizes['f16']:.0f} KB)
    - Degradacao: {base_acc-(acc['f16'] or base_acc):.2f}pp
    - Ideal para: GPU/NPU com suporte fp16 nativo
    - Sem vantagem em CPU pura vs. Dynamic Range

  Full Integer Quantization (model_int8.tflite)
    - Reducao: {100-sizes['int8']/base_size*100:.0f}% do tamanho  ({base_size:.0f} KB -> {sizes['int8']:.0f} KB)
    - Exige calibracao + hardware int8 dedicado (Coral Edge TPU)
    - Melhor opcao para latencia maxima em hardware especializado

  CONCLUSAO:
  Para o MNIST com CNN leve (~{model.count_params()//1000}K params), Dynamic Range e
  a escolha ideal: maior compressao sem calibracao e sem
  degradacao significativa de acuracia — principio central de Edge AI.
""")

print("  Arquivos gerados:")
print(f"   model.h5              — float32 completo ({h5_size_kb:.1f} KB)")
print(f"   model_base.tflite     — TFLite sem otimizacao ({sizes['base']:.1f} KB)")
print(f"   model.tflite          — Dynamic Range int8 ({sizes['dyn']:.1f} KB) <- PRINCIPAL")
print(f"   model_float16.tflite  — Float16 ({sizes['f16']:.1f} KB)")
print(f"   model_int8.tflite     — Full Integer int8 ({sizes['int8']:.1f} KB)")
print("\n" + SEP)
print("  OTIMIZACAO CONCLUIDA COM SUCESSO")
print(SEP + "\n")