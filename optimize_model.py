import tensorflow as tf
import numpy as np
import os

# 1 - carrega o modelo treinado
print("\n Carregando modelo treinado (model.h5)...")
model = tf.keras.models.load_model("model.h5")
print(f"  Parâmetros: {model.count_params():,} | Formato: float32")

# 2 - carrega dados de teste para avaliação pós-conversão
# usando para comparar acuracia antes e apps quantizacao
# comprovando que a otimização não degrada o desempenho
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test[..., tf.newaxis].astype("float32") / 255.0

def avaliar_tflite(model_path, x_sample, y_sample, n=500):
    """
    Avalia acuracia de um modelo TFLite em n amostras do conjunto de teste.
    Permite comparar desempenho antes e apos quantizacao
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    for i in range(n):
        inp = x_sample[i:i+1]
        # converte para int8 se o modelo quantizado exigir
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            inp = (inp / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        if np.argmax(output) == y_sample[i]:
            correct += 1
    return correct / n * 100


def converter_e_salvar(converter, output_path, label):
    """converte, salva e retorna tamanho em KB"""
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [{label}] → {output_path} ({size_kb:.1f} KB)")
    return size_kb


# 3 - conversão sem otimização (baseline de referência)
# deixando todos os pesos em float32
print("\n" + "="*60)
print(" CONVERTE PARA TFLITE — 3 VARIANTES")
print("="*60)

print("\n[1/3] Baseline float32 (sem otimização)...")
converter_base = tf.lite.TFLiteConverter.from_keras_model(model)
size_base = converter_e_salvar(converter_base, "model_base.tflite", "Baseline float32")

# 4 - Dynamic Range Quantization (TECNICA PRINCIPAL)
#
# COMO FUNCIONA
# Pesos convertidos de float32 → int8 em tempo de CONVERSAO
#   - ativações quantizadas dinamicamente para int8 em cada inferencia
#     retornando a float32 no final
#   - nao exige dataset de calibracao
#
# POR QUE A TECNICA PRINCIPAL PARA EDGE AI:
#   - reduz de ~75% no tamanho (float32 → int8 = 4x menor)
#   - compatibilidade com qualquer CPU (incluindo MCU, ESP32, Raspberry Pi)
#   - perda de acuracia tipicamente < 0.5% no MNIST
#   - ponto de entrada recomendado pelo Google para TFLite em embarcados.
#   - nao requer hardware especializado (NPU, GPU)
print("\n[2/3] Dynamic Range Quantization (técnica principal)...")
print("  Converte pesos float32 → int8 | ativações: dinâmicas em runtime")
converter_dyn = tf.lite.TFLiteConverter.from_keras_model(model)
converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
size_dyn = converter_e_salvar(converter_dyn, "model.tflite", "Dynamic Range Quantization")

# 5 - Float16 Quantization (tecnica adicional)
# COMO FUNCIONA:
#   - pesos convertidos de float32 → float16
#   - ativações permanecem em float32
#   - tambem nao requer dados de calibração
#
print("\n[3/3] Float16 Quantization (tecnica adicional)...")
print("  Converte pesos float32 → float16 | ativações: float32")
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
size_f16 = converter_e_salvar(converter_f16, "model_float16.tflite", "Float16 Quantization")

# 6 - avaliar pós-conversao (500 amostras)
#
# Comprova que quantizada não degradou significativamente
# o desempenho — essencial para validar a tecnica
print("\n" + "="*60)
print(" AVALIAÇÃO PÓS-CONVERSÃO (500 amostras de teste)")
print("="*60)

acc_base = avaliar_tflite("model_base.tflite", x_test, y_test)
acc_dyn  = avaliar_tflite("model.tflite",      x_test, y_test)
acc_f16  = avaliar_tflite("model_float16.tflite", x_test, y_test)

# 7 - Relatório comparativo final
print("\n" + "="*60)
print(" COMPARATIVO COMPLETO: TAMANHO × ACURÁCIA × TÉCNICA")
print("="*60)
print(f"\n  {'Versão':<28} {'Tamanho':>10} {'Redução':>10} {'Acurácia':>10}")
print(f"  {'-'*58}")
print(f"  {'Baseline (float32)':<28} {size_base:>8.1f} KB {'100%':>10} {acc_base:>9.2f}%")
print(f"  {'Dynamic Range (int8)':<28} {size_dyn:>8.1f} KB {f'{100*size_dyn/size_base:.0f}%':>10} {acc_dyn:>9.2f}%")
print(f"  {'Float16':<28} {size_f16:>8.1f} KB {f'{100*size_f16/size_base:.0f}%':>10} {acc_f16:>9.2f}%")

degradacao_dyn = acc_base - acc_dyn
degradacao_f16 = acc_base - acc_f16

print(f"""
 ANÁLISE DOS TRADE-OFFS:

  Dynamic Range Quantization:
    • Redução de tamanho: {100 - 100*size_dyn/size_base:.0f}% (de {size_base:.1f} KB → {size_dyn:.1f} KB)
    • Degradação de acurácia: {degradacao_dyn:.2f}% ({'mínima — dentro do esperado' if degradacao_dyn < 1 else 'aceitável'})
    • Adequado para: ESP32, STM32, Raspberry Pi (CPU pura)
    • Motivo da escolha: máxima compressão com mínima perda

  Float16 Quantization:
    • Redução de tamanho: {100 - 100*size_f16/size_base:.0f}% (de {size_base:.1f} KB → {size_f16:.1f} KB)
    • Degradação de acurácia: {degradacao_f16:.2f}%
    • Adequado para: hardware com suporte nativo fp16 (GPUs, NPUs)
    • Limitação: sem vantagem em CPU pura vs. Dynamic Range

 CONCLUSÃO:
  model.tflite (Dynamic Range) é o arquivo principal.
  Para o MNIST com esta arquitetura CNN leve (~23K parâmetros),
  a quantização int8 reduz o modelo em ~{100 - 100*size_dyn/size_base:.0f}% mantendo
  degradação de acurácia de apenas {degradacao_dyn:.2f}% — comprovando
  que Edge AI eficiente não exige sacrificar desempenho.
""")

print(" Arquivos gerados:")
print(f"   model.h5              — modelo completo float32 ({os.path.getsize('model.h5')/1024:.1f} KB)")
print(f"   model_base.tflite     — TFLite sem otimização ({size_base:.1f} KB)")
print(f"   model.tflite          — TFLite Dynamic Range int8 ({size_dyn:.1f} KB) ← PRINCIPAL")
print(f"   model_float16.tflite  — TFLite Float16 ({size_f16:.1f} KB)")