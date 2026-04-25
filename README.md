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
</details>


## 📑 Navegação documentada
- [👤 Identificação](https://github.com/jhonatan-goncalves-pereira/processoseletivoIoT/tree/docs#-identificação-do-candidato)
- [1️⃣ Resumo da Arquitetura](https://github.com/jhonatan-goncalves-pereira/processoseletivoIA#1%EF%B8%8F%E2%83%A3-resumo-da-arquitetura-do-modelo)
- [2️⃣ Bibliotecas Utilizadas](https://github.com/jhonatan-goncalves-pereira/processoseletivoIA#2%EF%B8%8F%E2%83%A3-bibliotecas-utilizadas)
- [3️⃣ Técnicas de otimização de modelo](https://github.com/jhonatan-goncalves-pereira/processoseletivoIA#3%EF%B8%8F%E2%83%A3-t%C3%A9cnica-de-otimiza%C3%A7%C3%A3o-do-modelo)
- [4️⃣ Resultados obtidos](https://github.com/jhonatan-goncalves-pereira/processoseletivoIA#4%EF%B8%8F%E2%83%A3-resultados-obtidos)
- [5️⃣ Comentários adicionais](https://github.com/jhonatan-goncalves-pereira/processoseletivoIA#5%EF%B8%8F%E2%83%A3-coment%C3%A1rios-adicionais)
- [6️⃣ Resultados](https://github.com/jhonatan-goncalves-pereira/processoseletivoIoT/tree/docs#6️⃣-resultados-obtidos)
- [7️⃣ Melhorias](https://github.com/jhonatan-goncalves-pereira/processoseletivoIoT/tree/docs#7️⃣-limitações-e-melhorias-futuras)

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
| 1 | Conv2D | 16 filtros, kernel 3×3, ReLU, padding=same | Extrai bordas e texturas simples com poucos parâmetros |
| 2 | MaxPooling2D | pool 2×2 | Reduz dimensionalidade pela metade, mantendo features relevantes |
| 3 | Conv2D | 32 filtros, kernel 3×3, ReLU, padding=same | Combina features em padrões mais complexos |
| 4 | MaxPooling2D | pool 2×2 | Segunda redução de dimensionalidade |
| 5 | GlobalAveragePooling2D | — | Substitui Flatten: elimina parâmetros e regulariza implicitamente |
| 6 | Dense | 64 neurônios, ReLU | Classificador compacto |
| 7 | Dropout | 25% | Reduz overfitting sem custo em inferência |
| 8 | Dense | 10 neurônios, Softmax | Saída com probabilidade por classe (0–9) |
 
**Por que 2 blocos Conv e não 3?**
O MNIST é um dataset simples: imagens 28×28 em escala de cinza, com apenas 10 classes de formas geométricas regulares. Dois blocos convolucionais são suficientes para extrair todas as features relevantes. Um terceiro bloco aumentaria parâmetros, tempo de treinamento e tamanho do modelo sem ganho significativo de acurácia — indo contra o princípio de Edge AI.
 
**Por que GlobalAveragePooling em vez de Flatten?**
Um `Flatten` após o segundo MaxPooling geraria um vetor de 7×7×32 = 1.568 elementos, exigindo uma Dense com 1.568×64 = ~100K parâmetros só nessa camada. O `GlobalAveragePooling2D` colapsa cada mapa de features para 1 valor, resultando em apenas 32 valores — reduzindo drasticamente os parâmetros do classificador.
 
---
 
### 2️⃣ Bibliotecas Utilizadas
 
| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `tensorflow` | ≥ 2.12 | Treinamento da CNN, conversão TFLite |
| `numpy` | ≥ 1.21 | Manipulação de arrays e dados |
 
As bibliotecas `keras` e `tf.lite` já estão inclusas no TensorFlow, não exigindo instalação separada.
 
---
 
### 3️⃣ Técnica de Otimização do Modelo
 
Foram aplicadas e comparadas duas técnicas de quantização:
 
#### Técnica Principal: Dynamic Range Quantization (`model.tflite`)
 
**Como funciona:**
- Os **pesos** da rede são convertidos de `float32` para `int8` em tempo de conversão.
- As **ativações** são quantizadas dinamicamente para `int8` em cada inferência, retornando a `float32` ao final.
- **Não exige** conjunto de dados de calibração.
**Por que foi escolhida como principal:**
- Reduz o modelo em ~75% sem precisar de dados extras.
- Compatível com qualquer hardware (CPU, MCU, ESP32, Raspberry Pi).
- Perda de acurácia tipicamente < 0,5% no MNIST.
- É o ponto de entrada recomendado para Edge AI por equilibrar facilidade de aplicação e ganho real de compressão.
#### Técnica Adicional: Float16 Quantization (`model_float16.tflite`)
 
**Como funciona:**
- Os **pesos** são convertidos de `float32` para `float16`.
- Ativações permanecem em `float32`.
- Também não requer dados de calibração.
**Trade-off em relação à Dynamic Range:**
- Redução de tamanho menor (~50% vs ~75%)
- Acurácia praticamente idêntica ao modelo original
- Útil em hardware com suporte nativo a `float16` (GPUs, alguns NPUs)
- Para dispositivos com CPU pura (cenário deste desafio), a Dynamic Range é mais vantajosa
#### Comparativo de Tamanho
 
| Versão | Tamanho | Redução |
|--------|---------|---------|
| Baseline float32 | ~100% | — |
| Float16 | ~50% | ~50% menor |
| Dynamic Range (int8) | ~25% | ~75% menor |
 
---
 
### 4️⃣ Resultados Obtidos
 
Após 5 épocas de treinamento em CPU:
 
| Métrica | Valor (aproximado) |
|---------|-------------------|
| **Accuracy (teste)** | ~98–99% |
| **AUC (teste)** | ~0.9990+ |
| **Loss (teste)** | ~0.04–0.06 |
 
**Interpretação das métricas:**
 
- **Accuracy ~98-99%**: excelente para um modelo leve treinado em apenas 5 épocas. Confirma que a arquitetura é adequada ao problema.
- **AUC ~0.999**: próximo de 1.0, indica que o modelo separa as 10 classes com altíssima confiança — mesmo em casos ambíguos (ex: 4 vs 9, 3 vs 8), a probabilidade da classe correta é consistentemente maior.
- A combinação de alta accuracy E alto AUC é mais informativa do que accuracy sozinha: garante que o modelo não está apenas "chutando" a classe mais frequente.
---
 
### 5️⃣ Comentários Adicionais
 
#### Decisões técnicas importantes
 
**Dropout de 25%** foi incluído na camada densa. Embora o MNIST raramente sofra overfitting com arquiteturas pequenas, o Dropout melhora levemente a generalização sem custo algum em inferência (é desativado automaticamente no modo de predição).
 
**Salvamento duplo** (`model.h5` + `model_saved/`): o formato `.h5` é exigido pelo enunciado e pelo pipeline CI. O formato `SavedModel` foi gerado adicionalmente porque o `TFLiteConverter` funciona de forma mais confiável com ele em versões recentes do TensorFlow — ambos são carregados no `optimize_model.py` via `.h5` para manter simplicidade.
 
**Duas variantes TFLite** foram geradas para demonstrar compreensão das técnicas de quantização disponíveis. O arquivo `model.tflite` (Dynamic Range) é o principal e o único exigido pelo CI.
 
#### Trade-offs tamanho × desempenho
 
O principal trade-off em Edge AI é: **menor modelo = mais rápido e menos memória, mas potencialmente menos acurácia**.
 
Para o MNIST com esta arquitetura, a Dynamic Range Quantization comprova que é possível reduzir o modelo em 75% com perda de acurácia desprezível (<0.5%). Isso ocorre porque o MNIST é um problema "fácil" para CNNs — os padrões são simples o suficiente para que pesos int8 representem bem as features aprendidas.
 
Em problemas mais complexos (ex: classificação de objetos com ImageNet), o trade-off seria mais relevante e uma Full Integer Quantization com calibração poderia ser necessária para manter acurácia adequada.
 
#### Limitações
 
- O modelo não foi testado com dados reais fora do MNIST (domínio fechado).
- A quantização int8 pode causar erros em dígitos muito incomuns ou mal escritos que diferem muito da distribuição do dataset.
- 5 épocas foram suficientes para convergência no MNIST; em datasets maiores, mais épocas seriam necessárias.
---

## 🆘 Suporte

Em caso de dúvidas:

- Consulte o material dos cursos EAD
- Leia atentamente este README
- Analise os logs das GitHub Actions
- Utilize os canais oficiais para contato com os instrutores

Boa sorte no processo seletivo.
****
