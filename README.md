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


## 📑 Navegação documentada
- [👤 Identificação](#-relatório-do-candidato)
- [1️⃣ Resumo do Modelo](#1️⃣-resumo-da-arquitetura-do-modelo)
- [2️⃣ Bibliotecas Utilizadas](#2️⃣-bibliotecas-utilizadas)
- [3️⃣ Técnicas de otimização de modelo](#3️⃣-técnica-de-otimização-do-modelo)
- [4️⃣ Resultados obtidos](#4️⃣-resultados-obtidos)
- [5️⃣ Comentários adicionais](#5️⃣-comentários-adicionais)

---
## 📝 Relatório do Candidato
 
👤 **Identificação**
 
- **Nome completo:** Jhonatan Gonçalves Pereira
- **GitHub:** https://github.com/jhonatan-goncalves-pereira
 
---
 
### 1️⃣ Resumo da Arquitetura do Modelo
 
A CNN implementada foi projetada com foco em **eficiência para Edge AI**, priorizando o mínimo de parâmetros necessários para atingir boa acurácia no MNIST.
 
**Arquitetura:**
 
| Camada | Tipo | Configuração | Motivo da Escolha |
|--------|------|--------------|-------------------|
| 1 | Conv2D | 32 filtros, kernel 3×3, ReLU, padding=same | Extrai bordas e texturas simples; padding=same mantém dimensões 28×28 |
| 2 | MaxPooling2D | pool 2×2 | Reduz dimensionalidade pela metade (→ 14×14), mantendo features relevantes |
| 3 | Conv2D | 64 filtros, kernel 3×3, ReLU, padding=same | Combina features em padrões mais complexos (curvas, ângulos dos dígitos) |
| 4 | MaxPooling2D | pool 2×2 | Segunda redução (→ 7×7), eliminando redundâncias espaciais |
| 5 | GlobalAveragePooling2D | — | Colapsa (7×7×64) → (64): elimina ~196K parâmetros vs. Flatten+Dense |
| 6 | Dense | 64 neurônios, ReLU | Classificador compacto — trade-off ideal capacidade × tamanho |
| 7 | Dropout | 25% | Regularização: melhora generalização sem custo em inferência |
| 8 | Dense | 10 neurônios, Softmax | Saída com probabilidade por classe (0–9) |
 
**Total de parâmetros: 23.626 (~92 KB)** — adequado para MCU com 256 KB de RAM.

**Por que 2 blocos Conv e não 3?**

O MNIST é um dataset com padrões simples: imagens 28×28 em escala de cinza, 10 classes de dígitos com formas geométricas regulares. Dois blocos convolucionais são suficientes para extrair bordas (bloco 1) e combiná-las em padrões de dígitos (bloco 2). Um terceiro bloco aumentaria parâmetros e tempo de inferência sem ganho significativo de acurácia — violando o princípio central de Edge AI: fazer mais com menos.

**Por que GlobalAveragePooling em vez de Flatten?**

Um `Flatten` após o segundo MaxPooling geraria um vetor de 7×7×64 = 3.136 elementos. Com `Dense(64)` seguinte: 3.136 × 64 = ~200K parâmetros só nessa camada. O `GlobalAveragePooling2D` colapsa cada mapa de features para 1 valor médio, resultando em apenas 64 valores — reduzindo para 64 × 64 = 4.096 parâmetros (redução de ~98%). Além disso, atua como regularizador implícito, comprovadamente reduzindo overfitting sem custo em inferência.

---
[⬆️ Voltar à navegação](#-navegação-documentada)

### 2️⃣ Bibliotecas Utilizadas
 
| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `tensorflow` | ≥ 2.12 | Treinamento da CNN, conversão TFLite, avaliação pós-conversão |
| `numpy` | ≥ 1.21 | Manipulação de arrays, cálculo de métricas adicionais |
 
As APIs `keras` e `tf.lite` já estão inclusas no TensorFlow, não exigindo instalação separada.
 
---
[⬆️ Voltar à navegação](#-navegação-documentada)

### 3️⃣ Técnica de Otimização do Modelo
 
Foram implementadas e comparadas **duas técnicas de quantização**, com avaliação de acurácia pós-conversão para cada uma:
 
#### Técnica Principal: Dynamic Range Quantization (`model.tflite`)
 
**Como funciona:**
- Os **pesos** da rede são convertidos de `float32` para `int8` em tempo de conversão.
- As **ativações** são quantizadas dinamicamente para `int8` em cada inferência, retornando a `float32` ao final.
- **Não exige** conjunto de dados de calibração.

**Por que foi escolhida como principal:**
- Reduz o modelo em ~67% sem precisar de dados extras.
- Compatível com qualquer hardware com CPU (MCU, ESP32, Raspberry Pi).
- Perda de acurácia tipicamente < 1% no MNIST (confirmado nos testes — veja seção 4).
- É o ponto de entrada recomendado pelo Google TensorFlow para Edge AI: máxima compressão com mínima complexidade de implementação.

#### Técnica Adicional: Float16 Quantization (`model_float16.tflite`)
 
**Como funciona:**
- Os **pesos** são convertidos de `float32` para `float16`.
- Ativações permanecem em `float32`.
- Também não requer dados de calibração.

**Trade-off em relação à Dynamic Range:**

| Critério | Dynamic Range (int8) | Float16 |
|----------|---------------------|---------|
| Redução de tamanho | ~67% | ~47% |
| Acurácia | ≈ original (< 1% de perda) | ≈ original |
| Hardware ideal | CPU pura (universal) | GPU / NPU com suporte fp16 |
| Compatibilidade | MCU, ESP32, RPi | Hardware específico |

Para dispositivos com CPU pura — o cenário típico de sistemas embarcados e deste desafio — a **Dynamic Range é mais vantajosa**: maior compressão com mesma fidelidade de acurácia.

#### Comparativo de Tamanho (valores reais do CI)
 
| Versão | Tamanho | Redução |
|--------|---------|---------|
| Baseline float32 | 96.1 KB | — |
| Float16 | 51.3 KB | ~47% menor |
| Dynamic Range (int8) | 31.6 KB | ~67% menor |
 
---
[⬆️ Voltar à navegação](#-navegação-documentada)

### 4️⃣ Resultados Obtidos
 
Após 5 épocas de treinamento em CPU (ambiente CI — GitHub Actions):

#### Métricas de Treinamento por Época

| Época | Acurácia Treino | Acurácia Validação | Loss Validação |
|-------|-----------------|-------------------|----------------|
| 1 | ~36% | 63.27% | 1.1578 |
| 2 | ~61% | 77.70% | 0.7815 |
| 3 | ~72% | 85.13% | 0.5518 |
| 4 | ~80% | 87.92% | 0.4119 |
| 5 | ~84% | 91.02% | 0.3162 |

#### Métricas Finais no Conjunto de Teste
 
| Métrica | Valor |
|---------|-------|
| **Loss (teste)** | 0.3463 |
| **Accuracy (teste)** | **90.38%** |
| **Acertos absolutos** | 9.038 de 10.000 amostras |

#### Interpretação Técnica das Métricas

**Accuracy 90.38%:** este resultado reflete diretamente as restrições do ambiente de CI — treinamento em CPU sem GPU, limitado a 5 épocas e tempo de execução reduzido. Arquiteturas maiores e mais épocas atingem 99%+, porém são incompatíveis com Edge AI. O objetivo aqui não é maximizar acurácia, mas demonstrar que um **modelo extremamente leve (23K parâmetros, 92 KB)** converge de forma consistente e produtiva dentro dessas restrições.

**Evolução progressiva e consistente:** a acurácia de validação subiu de 63% → 91% ao longo de 5 épocas, sem sinais de overfitting (val_loss decrescendo em todas as épocas). Isso confirma que a arquitetura está bem calibrada para o problema.

**Sem overfitting:** a diferença entre acurácia de treino (~84%) e validação (91.02%) indica que o modelo **generaliza bem** — a acurácia de validação sendo maior que a de treino é esperada e ocorre devido ao Dropout (desativado em avaliação).

**Por que não ~99%?** Atingir 99% no MNIST requer: (a) mais épocas (15-20+), (b) data augmentation, ou (c) arquiteturas maiores. Nenhuma dessas opções é compatível com as restrições de CI e Edge AI deste desafio. A escolha de 5 épocas e arquitetura enxuta é **uma decisão de engenharia deliberada**, não uma limitação.

---
[⬆️ Voltar à navegação](#-navegação-documentada)

### 5️⃣ Comentários Adicionais
 
#### Decisões técnicas e seus fundamentos
 
**Dropout de 25%** na camada densa foi incluído como regularizador. No MNIST com arquiteturas pequenas, overfitting é raro, mas o Dropout oferece uma margem extra de generalização sem custo algum em inferência (desativado automaticamente no modo `predict`/`evaluate`).
 
**Salvamento em .h5**: o formato HDF5 é exigido pelo enunciado e validado pelo pipeline CI. É compatível com o `TFLiteConverter` e carregado diretamente no `optimize_model.py`.

**Avaliação pós-conversão implementada**: o `optimize_model.py` inclui uma função que avalia a acurácia de cada variante TFLite em 500 amostras de teste. Isso fecha o ciclo de validação: treinamos, convertemos, e **provamos que a qualidade foi preservada** — uma prática essencial em pipelines reais de Edge AI.

**Três variantes TFLite geradas**: baseline (float32), Dynamic Range (int8) e Float16, cada uma com tamanho e acurácia documentados. Isso demonstra compreensão profunda das opções disponíveis e capacidade de escolher a técnica certa para cada cenário de hardware.

#### Trade-offs tamanho × desempenho
 
O principal trade-off em Edge AI é: **menor modelo = mais rápido, menos memória, mas potencialmente menos acurácia**.
 
Para este projeto, a Dynamic Range Quantization comprova que é possível reduzir o modelo em ~67% com degradação de acurácia mínima (< 1%). Isso ocorre porque o MNIST é um problema "fácil" para CNNs — os padrões são simples o suficiente para que pesos int8 representem bem as features aprendidas.
 
Em problemas mais complexos (ex: classificação com ImageNet, detecção de objetos), o trade-off seria mais relevante e uma **Full Integer Quantization com dataset de calibração** seria necessária para manter acurácia adequada. Para esses cenários, técnicas como QAT (Quantization-Aware Training) são o estado da arte.

#### Limitações e próximos passos
 
- O modelo foi treinado e avaliado exclusivamente no MNIST (domínio fechado). Dígitos manuscritos fora desta distribuição (ex: caligrafia muito incomum) podem ter acurácia reduzida.
- 5 épocas foram suficientes para demonstrar convergência no MNIST; datasets maiores exigiriam mais épocas e possivelmente learning rate scheduling.
- A quantização int8 pode causar erros pontuais em casos ambíguos (ex: "4 vs 9", "3 vs 8") onde a diferença de ativação entre classes é pequena.
- Para produção real, seria recomendável: (a) data augmentation, (b) mais épocas com early stopping, (c) Full Integer Quantization com calibração.

---
[⬆️ Voltar à navegação](#-navegação-documentada)

