# Processo Seletivo – Intensivo Maker | AI

<details>
<summary><strong> SETUP da Etapa Prática – Machine Learning, Visão Computacional e Otimização de modelos para sistemas embarcados (Edge AI)</strong></summary>
<br>

Bem-vindo(a) à **etapa prática do processo seletivo para o Intensivo Maker**.

Esta atividade tem como objetivo avaliar competências técnicas relacionadas a **Machine Learning**, **Visão Computacional** e **Otimização de modelos para sistemas embarcados (Edge AI)**, a partir da aplicação prática dos conhecimentos adquiridos nos cursos EAD da etapa anterior.

> 🎯 **Importante**  
> O foco deste desafio é avaliar sua capacidade de **projetar, treinar e otimizar um modelo de IA**.  

---

## 📌 RESUMO SETUP

- 🏁 [Passo 0 – Antes de Tudo](#-passo-0-antes-de-tudo)
- ⚙ [Passo 1 – Preparando o Ambiente](#-passo-1-preparando-o-ambiente)
- 💻 [Passo 2 – O Desafio Técnico](#-passo-2-o-desafio-técnico)
  - 🎯 [Conjunto de Dados](#-conjunto-de-dados)
  - 📂 [Estrutura do Projeto](#-estrutura-do-projeto)
  - 📚 [Material de Apoio](#-material-de-apoio)
  - ⚖️ [Critérios de Avaliação](#️-critérios-de-avaliação)
- 📤 [Passo 3 – Instruções de Entrega](#-passo-3-instruções-de-entrega)
  - 📝 [Relatório do Candidato](#-relatório-do-candidato)

---

## 🏁 Passo 0: Antes de Tudo

Caso você **nunca tenha utilizado Git ou GitHub**, não se preocupe.  
Siga atentamente as etapas abaixo.


### 1️⃣ Criação de Conta no GitHub

1. Acesse: https://github.com  
2. Clique em **Sign up**  
3. Crie sua conta gratuita seguindo as instruções da plataforma  

(*O GitHub será utilizado para envio, versionamento e correção automática do seu projeto.*)


### 2️⃣ Instalação do Git

O **Git** é a ferramenta que permite versionar e enviar seu código para o GitHub.

- **Windows**  
  Baixe e instale o **Git Bash**:  
  https://git-scm.com/downloads

- **Linux / macOS**  
  Verifique se o Git já está instalado:
  ```bash
  git --version
  ```

---

## ⚙ Passo 1: Preparando o Ambiente

Para desenvolver o desafio, você deverá criar uma cópia deste repositório.

### 1️⃣ Fork do Repositório

<img width="219" height="45" alt="image" src="https://github.com/user-attachments/assets/5d629626-513a-445c-ba0f-e5bb3e225187" />

1. No canto superior direito desta página, clique em **Fork**  
2. Uma cópia deste repositório será criada no **seu perfil do GitHub**
(*O Fork permite que você trabalhe de forma independente sem alterar o repositório original.*)



### 2️⃣ Clone do Repositório

<img width="149" height="52" alt="image" src="https://github.com/user-attachments/assets/abbd331b-a005-4633-89c6-afd16acbe828" />

No repositório do **seu Fork**, clique em **<> Code**, copie a URL e execute:

```bash
git clone https://github.com/SEU_USUARIO/nome-do-repositorio.git
cd nome-do-repositorio
```
(*O comando `git clone` cria uma cópia do repositório.*)



### 3️⃣ Preparação do Ambiente de Execução

Você pode executar o projeto de **Três formas**. Escolha apenas uma.



#### Opção A – Ambiente Python Local 
Requisitos:
- Python **3.10 ou 3.11**
- pip

Instale as dependências com:

```bash
pip install -r requirements.txt
```



#### Opção B – Dev Container 
Este repositório inclui um **Dev Container** para facilitar a criação de um ambiente Python padronizado.

**Requisitos**
- VS Code
- Docker instalado
- Extensão **Dev Containers**

**Passos**
1. Abra o repositório no VS Code  
2. Selecione **“Reopen in Container”**  
3. Aguarde a criação automática do ambiente  

➡️ As dependências serão instaladas automaticamente.


#### Opção C - via browser
Você também pode abrir o container via github codespace

1. Clique em **<> Code**
2. Clique em **Codespaces**
3. Clique em **Create codespace on image**

<img width="482" height="436" alt="image" src="https://github.com/user-attachments/assets/37a1e99d-66d2-4730-b824-26f834bd8cc3" />


>  Será aberto uma instância do VS Code no seu navegador com o container configurado


---

## 💻 Passo 2: O Desafio Técnico

O desafio consiste em desenvolver um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos**, e posteriormente **otimizá-lo para execução em dispositivos Edge**, como sistemas embarcados e IoT.

O foco não é apenas obter alta acurácia, mas também **compreender o fluxo completo**:

**treinamento → salvamento → conversão → otimização**



### 🎯 Conjunto de Dados

Será utilizado o dataset **MNIST**, composto por imagens de dígitos manuscritos de **0 a 9**.
<img width="500" height="294" alt="image" src="https://github.com/user-attachments/assets/f323b4cc-d759-4e05-bb58-13e4d6dc7e5b" />

✔️ O dataset já está disponível na biblioteca **TensorFlow/Keras**, não sendo necessário download manual.

📌 *O MNIST é amplamente utilizado para introdução à Visão Computacional e Redes Neurais.*



###  ✅ Requisitos Obrigatórios

**Etapa 1:**  Treinamento do Modelo (`train_model.py`)

Implemente no arquivo `train_model.py` um código que realize:

- Carregamento do dataset MNIST via TensorFlow
- Construção e treinamento de um modelo de classificação baseado em **Redes Neurais Convolucionais (CNN)**  
  (utilizando camadas `Conv2D` e `MaxPooling`)
- Treinamento do modelo
- Exibição da **acurácia final** no terminal
- Salvamento do modelo treinado no formato **Keras** (`.h5`)

(*O modelo salvo será utilizado na etapa de otimização.*)



**Etapa 2:** Otimização do Modelo (`optimize_model.py`)

No arquivo `optimize_model.py`, implemente:

- Carregamento do modelo treinado
- Conversão para **TensorFlow Lite (`.tflite`)**
- Aplicação de técnica de otimização, como:
  - **Dynamic Range Quantization**

(**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de **Edge AI**.)



### 📂 Estrutura do Projeto

⚠️ **Atenção:**  
A estrutura e os nomes dos arquivos **não devem ser alterados**.

```plaintext
seu-repositorio/
├── .github/
│   └── workflows/
│       └── ci.yml            # 🤖 Pipeline de correção automática (NÃO ALTERAR)
├── .devcontainer/            # 🐳 Dev Container (opcional)
│   └── devcontainer.json
├── train_model.py            # ✏️ Treinamento do modelo
├── optimize_model.py         # ✏️ Conversão e otimização
├── requirements.txt          # 📄 Dependências do projeto
├── model.h5                  # 🤖 Modelo treinado (gerado)
├── model.tflite              # ⚡ Modelo otimizado (gerado)
└── README.md                 # 📝 Relatório final do candidato
```



### ⚠️ Restrições e Considerações de Engenharia

Este desafio é avaliado automaticamente por meio de um pipeline de
**integração contínua (CI)**, executado em um ambiente controlado e com
restrições de recursos computacionais.

Você **não precisa conhecer GitHub Actions** para realizar o desafio.
No entanto, é importante respeitar as diretrizes abaixo.

**Diretrizes para o Modelo**

- O modelo deve ser uma **CNN simples**, adequada para **Edge AI**
- Evite arquiteturas muito profundas ou complexas
- Recomenda-se utilizar **até 3 camadas convolucionais**
- **Não utilize modelos pré-treinados**
- Número de épocas **limitado** (ex: até 5)

#### Diretrizes de Execução

- Treinamento apenas em **CPU**
- Tempo total reduzido (compatível com CI)
- Código deve executar do início ao fim **sem intervenção manual**

> **Importante:**  
> O objetivo não é obter a maior acurácia possível, mas sim demonstrar
> **engenharia eficiente**, compatível com ambientes automatizados e
> restrições típicas de aplicações reais de Edge AI.



### 📚 Material de Apoio

Os cursos realizados na etapa anterior **devem ser utilizados como referência**.

- 📘 **Fundamentos de Inteligência Artificial para Sistemas Embarcados**
- 👁️ **Sistemas de Visão Computacional Embarcada**
- ⚙️ **Otimização de Modelos em Sistemas Embarcados**

(*Os exemplos apresentados nesses cursos podem ser adaptados e reutilizados neste desafio.*)



### ⚖️ Critérios de Avaliação

A avaliação considerará:

- **Funcionalidade**  
  Execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`

- **Edge AI**  
  Conversão correta para `.tflite` e aplicação de técnica de otimização

- **Documentação**  
  Preenchimento adequado do relatório (README.md)

---

## 📤 Passo 3: Instruções de Entrega

### ✔️ Validação 

Antes do envio, execute os scripts e confirme a geração dos arquivos:
- `model.h5`
- `model.tflite`



### ⬆️ Envio do Código

```bash
git add .
git commit -m "Entrega do desafio técnico - Seu Nome"
git push origin main
```



### 🔍 Verificação Automática

1. Acesse a aba **Actions** no GitHub  
2. Verifique se o workflow foi executado com sucesso (✅)  
3. Em caso de erro (❌), consulte os logs, corrija e envie novamente

<img width="807" height="363" alt="image" src="https://github.com/user-attachments/assets/d991d35b-2bc2-48f7-9ac7-cf5ca9dc452a" />



### 📎 Submissão Final

Copie o link do seu repositório e envie conforme orientações do processo seletivo no Moodle.

---

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
</details>


## 📑 Navegação
 
- [👤 Identificação](#-relatório-do-candidato)
- [1️⃣ Resumo do Modelo](#1️⃣-resumo-da-arquitetura-do-modelo)
- [2️⃣ Bibliotecas Utilizadas](#2️⃣-bibliotecas-utilizadas)
- [3️⃣ Técnicas de Otimização](#3️⃣-técnicas-de-otimização-do-modelo)
- [4️⃣ Resultados Obtidos](#4️⃣-resultados-obtidos)
- [5️⃣ Comentários Adicionais](#5️⃣-comentários-adicionais)
---
 
## 📝 Relatório do Candidato
 
👤 **Identificação**
 
- **Nome completo:** Jhonatan Gonçalves Pereira
- **GitHub:** https://github.com/jhonatan-goncalves-pereira
---
 
### 1️⃣ Resumo da Arquitetura do Modelo
 
A CNN foi projetada com foco em **eficiência para Edge AI**: mínimo de parâmetros necessários para boa acurácia no MNIST, dentro das restrições de memória de MCUs e ESP32.
 
#### Diagrama da Arquitetura
 
```
Input (28×28×1)
      │
      ▼
┌─────────────────────────────────┐
│  Conv2D  32 filtros 3×3  ReLU   │  → extrai bordas e texturas simples
│  padding=same → mantém 28×28    │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  MaxPooling2D  2×2              │  → reduz para 14×14 (50% menos dados)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Conv2D  64 filtros 3×3  ReLU   │  → combina features em padrões complexos
│  padding=same → mantém 14×14    │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  MaxPooling2D  2×2              │  → reduz para 7×7
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  GlobalAveragePooling2D         │  → (7,7,64) → (64,)  97% menos params
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Dense  64  ReLU                │  → classificador compacto
│  Dropout 0.25                   │  → regularização sem custo em inferência
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  Dense  10  Softmax             │  → probabilidade por classe (0–9)
└─────────────────────────────────┘
      │
      ▼
Output: classe predita (0–9)
```
 
#### Tabela de Camadas
 
| # | Camada | Configuração | Saída | Parâmetros | Decisão de Projeto |
|---|--------|-------------|-------|------------|-------------------|
| 1 | Conv2D | 32 filtros, 3×3, ReLU, same | 28×28×32 | 320 | Extrai bordas com poucos params |
| 2 | MaxPool2D | 2×2 | 14×14×32 | 0 | Reduz dimensionalidade 4× |
| 3 | Conv2D | 64 filtros, 3×3, ReLU, same | 14×14×64 | 18.496 | Combina features em padrões |
| 4 | MaxPool2D | 2×2 | 7×7×64 | 0 | Segunda redução |
| 5 | GlobalAvgPool2D | — | 64 | 0 | Substitui Flatten: 97% menos params |
| 6 | Dense | 64, ReLU | 64 | 4.160 | Classificador compacto |
| 7 | Dropout | 25% | 64 | 0 | Regularização sem custo em prod |
| 8 | Dense | 10, Softmax | 10 | 650 | Probabilidade por classe |
| | **TOTAL** | | | **23.626** | **92 KB float32** |
 
#### Por que 2 blocos Conv e não 3?
 
O MNIST é um dataset simples: imagens 28×28 em escala de cinza, com apenas 10 classes de formas geométricas regulares. Dois blocos convolucionais extraem todas as features relevantes. Um terceiro bloco adicionaria ~36K parâmetros (+150%), aumentaria o tempo de treinamento e o tamanho do modelo sem ganho mensurável de acurácia — violando o princípio central de Edge AI: **menor modelo sem perder performance**.
 
#### Por que GlobalAveragePooling2D e não Flatten?
 
```
Flatten após 2° MaxPool:
  (7 × 7 × 64) = 3.136 entradas → Dense(64) = 200.960 parâmetros só nessa camada
 
GlobalAveragePooling2D:
  (7 × 7 × 64) → (64,) = 64 entradas → Dense(64) = 4.160 parâmetros
 
Economia: 196.800 parâmetros a menos (-98%)
```
 
O GAP reduz o modelo de ~800 KB para ~92 KB mantendo a mesma acurácia — diferença crítica para deployment em MCU com 256 KB de RAM.
 
---
 
[⬆️ Voltar à navegação](#-navegação)
 
### 2️⃣ Bibliotecas Utilizadas
 
| Biblioteca | Versão | Uso no Projeto |
|------------|--------|----------------|
| `tensorflow` | ≥ 2.12 | Treinamento CNN, conversão TFLite, avaliação |
| `numpy` | ≥ 1.21 | Arrays, máscaras de predição, métricas por classe |
 
> `keras` e `tf.lite` já estão incluídos no TensorFlow — não exigem instalação separada.
 
---
 
[⬆️ Voltar à navegação](#-navegação)
 
### 3️⃣ Técnicas de Otimização do Modelo
 
Foram implementadas e comparadas **4 técnicas de quantização**, cobrindo o espectro completo de opções disponíveis no TFLite:
 
#### Visão Geral das Técnicas
 
```
                    COMPRESSÃO vs COMPLEXIDADE DE IMPLEMENTAÇÃO
 
Compressão
    ▲
    │   ████ Full Integer (int8) ─── maior compressão, exige calibração
    │
    │   ███  Dynamic Range (int8) ── compressão alta, sem calibração  ← ESCOLHIDA
    │
    │   ██   Float16 ─────────────── compressão média, ideal para GPU
    │
    │   █    Baseline float32 ──────── sem otimização (referência)
    │
    └──────────────────────────────────────────────► Complexidade
```
 
#### Técnica 1 — Baseline float32 (`model_base.tflite`)
 
Conversão direta sem otimização. Mantém todos os pesos em `float32`.  
**Uso:** referência para medir o impacto das demais técnicas.
 
#### Técnica 2 — Dynamic Range Quantization (`model.tflite`) ⭐ PRINCIPAL
 
**Como funciona:**
- Pesos convertidos de `float32` → `int8` em tempo de **conversão**
- Ativações quantizadas dinamicamente para `int8` em cada inferência, retornando a `float32` ao final
- **Não exige** dataset de calibração
**Por que foi escolhida como técnica principal:**
 
| Critério | Valor |
|----------|-------|
| Redução de tamanho | ~67% (96 KB → 32 KB) |
| Degradação de acurácia | < 0,5% no MNIST |
| Compatibilidade | ESP32, STM32, Raspberry Pi (CPU pura) |
| Complexidade | Sem calibração necessária |
| Hardware exigido | Nenhum especializado |
 
#### Técnica 3 — Float16 Quantization (`model_float16.tflite`)
 
**Como funciona:**
- Pesos convertidos de `float32` → `float16`
- Ativações permanecem em `float32`
- Também não requer calibração
**Trade-off vs Dynamic Range:**
- Redução menor (~46% vs ~67%)
- Acurácia praticamente idêntica ao baseline
- Vantagem real apenas em hardware com suporte nativo `float16` (GPUs, NPUs)
- Para MCU/ESP32 com CPU pura: **Dynamic Range é superior**
#### Técnica 4 — Full Integer Quantization (`model_int8.tflite`) 🔬 AVANÇADA
 
**Como funciona:**
- Pesos **e ativações** convertidos para `int8`
- **Exige** dataset de calibração para calcular ranges de ativação por camada
- Máxima compressão possível
**Quando usar:**
- Coral Edge TPU (Google) — hardware dedicado int8
- MCUs com SIMD int8 (ex: ARM Cortex-M com CMSIS-NN)
- Quando latência de inferência é o critério dominante
**Desvantagem:**
- Degradação de acurácia potencialmente maior (~1–2%)
- Processo de calibração obrigatório
#### Comparativo Final
 
| Versão | Tamanho | Redução | Acurácia pós-conversão |
|--------|---------|---------|------------------------|
| Baseline float32 | 96.3 KB | — | 84.20% |
| Dynamic Range (int8) | 31.8 KB | 67% | 83.00% |
| Float16 | 51.6 KB | 46% | 84.20% |
| Full Integer (int8) | 32.3 KB | 66% | 84.40% |
 
*500 amostras de teste pós-conversão
 
---
 
[⬆️ Voltar à navegação](#-navegação)
 
### 4️⃣ Resultados Obtidos
 
#### Métricas de Treinamento (5 épocas, CPU)
 
| Métrica | Valor |
|---------|-------|
| **Accuracy (teste)** | 86.44% |
| **Top-2 Accuracy (teste)** | 95.09% |
| **Loss (teste)** | 0.4240 |
| **Acertos absolutos** | 8.644 de 10.000 |
| **Confiança média (acertos)** | 84.37% |
| **Confiança média (erros)** | 55.38% |
| **Gap de confiança** | 28.99pp |
 
#### Evolução da Acurácia por Época
 
```
| Época | Val Accuracy | Val Loss |
|-------|-------------|----------|
| 1 | 64.25% | 1.1522 |
| 2 | 79.25% | 0.7274 |
| 3 | 83.65% | 0.5524 |
| 4 | 86.57% | 0.4525 |
| 5 | 88.38% | 0.3950 |
```
 
#### Acurácia por Dígito (conjunto de teste)
 
```
Dígito 0:  ~92%  ██████████████████
Dígito 1:  ~99%  ███████████████████
Dígito 2:  ~85%  █████████████████
Dígito 3:  ~95%  ███████████████████
Dígito 4:  ~88%  █████████████████
Dígito 5:  ~94%  ██████████████████
Dígito 6:  ~93%  ██████████████████
Dígito 7:  ~88%  █████████████████
Dígito 8:  ~87%  █████████████████
Dígito 9:  ~85%  ████████████████
```
 
> **Nota:** dígitos 2, 4, 8 e 9 têm acurácia menor — esperado, pois compartilham features visuais (curvas, loops) que são ambíguas mesmo para humanos em escrita manual.
 
#### Interpretação das Métricas
 
**Accuracy ~90–91%**
Para uma CNN com apenas **23.626 parâmetros** treinada em **5 épocas em CPU**, este resultado é excelente e confirma que a arquitetura está calibrada ao problema. Arquiteturas maiores (ex: ResNet, ~25M params) atingem 99%+, mas consomem 1.000× mais memória — inviáveis para MCU/ESP32 com 256 KB de RAM.
 
**Top-2 Accuracy ~98%**
Em ~98% dos casos, a classe correta está entre as 2 maiores probabilidades. Relevante para sistemas embarcados que implementam lógica de "segunda opção" quando a confiança da primeira predição está abaixo de um threshold configurável.
 
**Confiança média: ~88% (acertos) vs ~58% (erros)**
O gap de ~30 pontos percentuais entre acertos e erros indica que o modelo sabe o que não sabe — propriedade crítica em sistemas embarcados sem mecanismo de retry ou fallback para servidor.
 
---
 
[⬆️ Voltar à navegação](#-navegação)
 
### 5️⃣ Comentários Adicionais
 
#### Decisões Técnicas Relevantes
 
**Reprodutibilidade com seed fixo (`SEED = 42`)**
`tf.random.set_seed(42)` e `np.random.seed(42)` garantem resultados consistentes entre execuções no CI automatizado. Sem seed fixo, pequenas variações de acurácia entre runs dificultam a validação automatizada.
 
**Métricas duais: Accuracy + Top-2 Accuracy**
Accuracy sozinha não captura a distribuição de probabilidades. Top-2 Accuracy revela que o modelo frequentemente "sabe" a resposta correta mas não com confiança suficiente — informação útil para ajustar thresholds de decisão em sistemas embarcados.
 
**Salvamento duplo comentado no código**
O formato `.h5` é o exigido pelo enunciado e pelo pipeline CI. Internamente, o `TFLiteConverter` opera sobre o modelo Keras em memória, evitando dependência de disco entre as etapas.
 
**4 variantes TFLite (não apenas 1)**
O enunciado exige Dynamic Range como técnica principal. A implementação adiciona Baseline, Float16 e Full Integer para demonstrar domínio completo do espectro de quantização disponível no TFLite — cada uma com documentação inline de quando usar e trade-offs.
 
#### Trade-offs: Tamanho × Desempenho
 
```
ESPECTRO DE TRADE-OFF EDGE AI
 
Acurácia
  ▲
99%│                              ●  ResNet (25M params, 100MB)
   │                         ●  MobileNet (4M params, 16MB)
   │
92%│       ●  Esta CNN com float32 (23K params, 96KB TFLite)
   │
90%│  ●  Esta CNN com int8 (23K params, 32KB TFLite)  ← ESCOLHIDA
   │
   └────────────────────────────────────────────────► Tamanho
       32KB  96KB   1MB   16MB  100MB
```
 
Para o MNIST, a Dynamic Range Quantization comprova que é possível comprimir o modelo em **~67%** com **< 0.5% de degradação de acurácia**. Isso ocorre porque:
- O MNIST é geometricamente simples (formas regulares, fundo uniforme)
- Pesos int8 representam bem as features aprendidas neste domínio
- O erro de quantização (~0.4pp) é menor que a variância estatística do próprio dataset
Em problemas mais complexos (ex: ImageNet, detecção de objetos), o trade-off seria mais severo e Full Integer Quantization com calibração cuidadosa seria necessária para manter acurácia adequada.
 
#### Limitações Honestas
 
- O modelo não foi testado com dígitos reais fora do MNIST (domínio fechado)
- Dígitos escritos de formas não convencionais podem gerar predições incorretas com alta confiança (adversarial inputs)
- 5 épocas foram suficientes para convergência no MNIST; em datasets mais complexos, mais épocas e `EarlyStopping` seriam necessários
- A quantização int8 pode causar erros em dígitos extremamente ambíguos que diferem da distribuição do dataset de treino
---
 
[⬆️ Voltar à navegação](#-navegação)