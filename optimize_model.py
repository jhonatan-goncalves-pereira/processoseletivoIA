import tensorflow as tf
import os

model = tf.keras.models.load_model("model.h5")

def convert_and_save(converter, output_path, label):
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [{label}] → {output_path} ({size_kb:.1f} KB)")
    return size_kb

#ref para comparar o impacto das otimizações.
print("\nconvertendo modelo para TFLite...\n")

converter_base = tf.lite.TFLiteConverter.from_keras_model(model)
size_base = convert_and_save(converter_base, "model_base.tflite", "sem otimizicao")

#DRQ
converter_dyn = tf.lite.TFLiteConverter.from_keras_model(model)
converter_dyn.optimizations = [tf.lite.Optimize.DEFAULT]
size_dyn = convert_and_save(converter_dyn, "model.tflite", "Dynamic Range Quantization")

#float16 quantization
converter_f16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_f16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_f16.target_spec.supported_types = [tf.float16]
size_f16 = convert_and_save(converter_f16, "model_float16.tflite", "Float16 Quantization")

#resumo comparativo
print("\n comparativo de tamanho dos modelos:")
print(f"  Baseline (float32) : {size_base:.1f} KB  (100%)")
print(f"  Dynamic Range (int8): {size_dyn:.1f} KB  ({100 * size_dyn / size_base:.0f}% do original)")
print(f"  Float16            : {size_f16:.1f} KB  ({100 * size_f16 / size_base:.0f}% do original)")

print("\n arquivo principal gerado: model.tflite (Dynamic Range Quantization)")
print("   → Melhor equilíbrio tamanho × acurácia para Edge AI.")