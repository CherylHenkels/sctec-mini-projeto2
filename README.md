# ✨ Que número é esse AI?

Mini-projeto de **classificação de dígitos manuscritos** desenvolvido no **Módulo 2 do Programa SCTEC**. O notebook utiliza o dataset **MNIST** para comparar três algoritmos de aprendizado de máquina — **Support Vector Machine (SVM)**, **Random Forest** e **Rede Neural (MLP)** — e também explora o comportamento do classificador diante de classes não vistas durante o treinamento e de imagens manuscritas próprias.

## 📌 Objetivo

O projeto tem como objetivo construir e avaliar um pipeline completo de classificação de imagens, passando por:

- análise exploratória do MNIST;
- divisão estratificada em treino, validação e teste;
- normalização dos pixels;
- treinamento e ajuste de hiperparâmetros de três classificadores;
- comparação por acurácia, precisão, recall e F1-score;
- análise por matriz de confusão;
- experimento com **classes ocultadas** durante o treinamento;
- inferência em imagens manuscritas externas ao MNIST.

## 🗂️ Dataset

O notebook utiliza o **MNIST (Modified National Institute of Standards and Technology)**, carregado diretamente pelo `scikit-learn` por meio de:

```python
fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
```

O conjunto possui:

- **70.000 imagens**;
- **10 classes**, correspondentes aos dígitos de `0` a `9`;
- imagens originais de **28 × 28 pixels**;
- **784 features** após a vetorização de cada imagem;
- intensidade de pixels entre **0 e 255**.

A primeira etapa do notebook verifica a dimensionalidade dos dados, a distribuição das classes e exibe uma grade com exemplos dos dez dígitos.

## ⚙️ Pipeline de pré-processamento

Os dados são divididos de forma **estratificada**, preservando aproximadamente a proporção de cada classe nos três subconjuntos:

| Conjunto | Amostras | Proporção |
|---|---:|---:|
| Treino | 49.000 | 70% |
| Validação | 7.000 | 10% |
| Teste | 14.000 | 20% |

Em seguida, os valores dos pixels são normalizados para o intervalo `[0, 1]`:

```python
X_train_norm = X_train / 255.0
X_val_norm = X_val / 255.0
X_test_norm = X_test / 255.0
```

O conjunto de teste permanece separado durante a seleção dos modelos e só é utilizado na avaliação final.

## 🤖 Modelos avaliados

### Support Vector Machine — SVM

Foram testadas combinações de:

- `C`: `0.1`, `1`, `10` e `100`;
- `kernel`: `rbf` e `linear`.

Melhor configuração na validação:

```text
C = 10
kernel = rbf
Accuracy = 0.98143
F1 = 0.98134
```

### Random Forest

Foram avaliados:

- `n_estimators`: `50`, `100`, `200` e `400`;
- `max_depth`: `5`, `10` e `20`.

Melhor configuração na validação:

```text
n_estimators = 200
max_depth = 20
Accuracy = 0.96671
F1 = 0.96651
```

### Rede Neural — MLP

Foram testadas diferentes arquiteturas de camadas ocultas e duas funções de ativação:

- `hidden_layer_sizes`: `(64,)`, `(128,)`, `(16,)`, `(128, 64)`, `(16, 128)` e `(16, 64)`;
- `activation`: `relu` e `tanh`.

Melhor configuração na validação:

```text
hidden_layer_sizes = (128, 64)
activation = relu
Accuracy = 0.94214
F1 = 0.94172
```



## 📊 Resultados no conjunto de teste

Os melhores modelos de cada família são salvos com `joblib` e posteriormente avaliados no conjunto de teste.

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| **SVM** | **0.983000** | **0.983004** | **0.983000** | **0.982991** |
| Random Forest | 0.965357 | 0.965404 | 0.965357 | 0.965347 |
| MLP | 0.942571 | 0.942539 | 0.942571 | 0.942461 |

No experimento registrado no notebook, o **SVM apresentou o melhor desempenho preditivo**. Entretanto, olhando para o  custo computacional: os testes de SVM demoraram muito mais do que os de Random Forest, tornando o **Random Forest uma alternativa com bom equilíbrio entre desempenho e custo de treinamento**.

As matrizes de confusão são utilizadas para analisar os erros de classificação. O notebook observa confusão recorrente entre os dígitos **4 e 9**.

## 🧪 Desafio: classes ocultadas e inferência OOD

O projeto inclui um experimento de **class masking**:

1. os dígitos `4` e `7` são removidos do conjunto de treino;
2. um novo Random Forest é treinado somente com as demais classes;
3. o modelo é testado exclusivamente em imagens dos dígitos `4` e `7`.

Como essas classes nunca foram vistas no treinamento, o classificador é obrigado a atribuir as imagens a alguma das classes conhecidas. No experimento do notebook:

- a acurácia, precisão, recall e F1 ficaram em `0` nesse teste;
- tanto `4` quanto `7` foram classificados predominantemente como `9`;
- secundariamente, o `4` foi associado ao `8` e o `7` ao `2`.

Essa etapa é usada para discutir o comportamento de classificadores diante de exemplos **fora da distribuição conhecida durante o treinamento (OOD)** e o problema de previsões incorretas com aparente confiança.

## ✍️ Inferência em imagens manuscritas próprias

O notebook também aplica o melhor Random Forest a imagens externas armazenadas na pasta `imagens/`.

O pré-processamento realizado com OpenCV inclui:

1. leitura da imagem em escala de cinza;
2. aplicação de um leve `GaussianBlur`;
3. binarização e inversão das cores;
4. detecção da região que contém o dígito;
5. recorte com *padding*;
6. redimensionamento para `20 × 20` pixels;
7. centralização em um canvas preto de `28 × 28`;
8. normalização para `[0, 1]`;
9. achatamento para um vetor com 784 posições;
10. classificação e visualização das probabilidades previstas.

Nos testes descritos no notebook, o modelo acertou **8 dos 10 dígitos manuscritos**, correspondendo a **80% de acurácia** nessa pequena amostra própria. Os dígitos `7` e `9` foram os mais difíceis nesse experimento.

## 📁 Estrutura do projeto

```text
.
├── main.ipynb
├── requirements.txt
├── README.md
├── models/
│   ├── best_model_svm.pkl
│   ├── best_model_rf.pkl
│   └── best_model_nn.pkl
└── imagens/
    ├── numero0.png
    ├── numero1.png
    ├── numero2.png
    └── ...
```



## 📦 Dependências

O notebook utiliza as seguintes bibliotecas:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Joblib
- Scikit-learn

Uma instalação possível é:

```bash
pip install -r requirements.txt
```


## ▶️ Como executar


### 1. Clone o repositório

```bash
git clone https://github.com/CherylHenkels/sctec-mini-projeto2.git
```

### 2. Acesse a pasta

```bash
cd sctec-projeto1
```

### 3. Crie um ambiente virtual

```bash
python -m venv .venv
```

### 4. Ative o ambiente virtual

Linux/macOS

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

### 5. Instale as dependências

```bash
pip install -r requirements.txt
```

### 6. Execute o notebook

Abra e execute o arquivo `main.ipynb`.

---

## 🔄 Fluxo do notebook

```text
MNIST
  ↓
Análise exploratória
  ↓
Divisão estratificada 70% / 10% / 20%
  ↓
Normalização [0, 1]
  ↓
┌──────────┬───────────────┬─────────────┐
│   SVM    │ Random Forest │     MLP     │
└──────────┴───────────────┴─────────────┘
  ↓
Seleção pela validação
  ↓
Avaliação no conjunto de teste
  ↓
Matrizes de confusão e comparação
  ↓
Experimento OOD sem classes 4 e 7
  ↓
Inferência em imagens manuscritas próprias
```

## 🏁 Conclusão

O mini-projeto implementa um fluxo completo de classificação de imagens com o MNIST, desde a exploração dos dados até a avaliação de modelos e testes em condições diferentes daquelas vistas durante o treinamento.

Entre os três algoritmos avaliados, o **SVM com kernel RBF e `C=10` obteve a maior acurácia no conjunto de teste (98,3%)**. O **Random Forest com 200 árvores e profundidade máxima 20** apresentou desempenho um pouco inferior, mas com custo de treinamento muito menor no ambiente utilizado no projeto. O experimento com classes ocultadas e os testes com imagens próprias complementam a análise ao mostrar limitações importantes de generalização dos classificadores.

## 🎥 Video de demonstração

TODO: trocar link

---

## 👩‍💻 Autores

Projeto desenvolvido por 
* **Cheryl Henkels** - [GitHub](https://github.com/CherylHenkels)
* **João Vitor Lovato** - [GitHub](https://github.com/joaovitorcl1000)