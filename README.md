# Chernomorets_nosql_2

Семантичний пошук за науковими статтями arXiv з використанням Pinecone, sentence-transformers та гібридного пошуку BM25 + RRF.

## Налаштування оточення

1. Клонувати репозиторій
2. Створити віртуальне середовище: `python -m venv venv`
3. Активувати: `venv\Scripts\activate`
4. Встановити залежності: `pip install -r requirements.txt`
5. Створити файл `.env`:
   ```
   PINECONE_API_KEY=your-api-key-here
   ```
6. Завантажити датасет з Kaggle:
   ```
   kaggle datasets download -d Cornell-University/arxiv --unzip
   ```
7. Запустити скрипти послідовно:
   ```bash
   python scripts/01_prepare_data.py
   python scripts/02_embed.py
   python scripts/03_load_to_pinecone.py
   python scripts/04_search.py
   python scripts/05_chunking.py
   python scripts/06_hybrid_search.py
   ```

## Структура проєкту

```
.
├── .env
├── .gitignore
├── requirements.txt
├── data/
│   └── arxiv_subset.parquet
├── embeddings/
│   └── embeddings.npy
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
└── README.md
```

---

## Вивід скриптів

### 01_prepare_data.py

```
Читаємо датасет: 10000it [00:00, 95625.88it/s]
Завантажено статей: 10000
Розподіл за категоріями (топ-10):
category
astro-ph              1838
hep-th                 680
hep-ph                 671
quant-ph               564
gr-qc                  350
cond-mat.mes-hall      307
cond-mat.str-el        292
cond-mat.mtrl-sci      291
cond-mat.stat-mech     271
math.AG                209
Name: count, dtype: int64
Розподіл за роками:
year
2007    10000
Name: count, dtype: int64
Збережено в data/arxiv_subset.parquet
```

### 02_embed.py

```
Завантажуємо модель allenai/specter2_base...
Генеруємо ембеддинги для 10000 текстів...
Batches: 100%|█████████████████████| 157/157 [12:50<00:00,  4.91s/it]
Кількість текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.000000
Збережено у embeddings/embeddings.npy
```

### 03_load_to_pinecone.py

```
Індекс arxiv-papers створено
Завантажуємо 10000 векторів у Pinecone...
100%|████████████████████████████████| 50/50 [00:38<00:00,  1.30it/s]
Векторів в індексі: 10000
```

### 04_search.py

```
============================================================
Семантичний пошук: 'teaching machines to recognize objects in pictures'
============================================================
#1 score=0.8288
Title: Capturing knots in polymers
Category: cond-mat.soft | Year: 2007.0

#2 score=0.8263
Title: Symbolic sensors : one solution to the numerical-symbolic interface
Category: physics.ins-det | Year: 2007.0

#3 score=0.8256
Title: The Mathematics
Category: math.HO | Year: 2007.0

#4 score=0.8170
Title: Modeling the field of laser welding melt pool by RBFNN
Category: physics.comp-ph | Year: 2007.0

#5 score=0.8146
Title: Why should anyone care about computing with anyons?
Category: quant-ph | Year: 2007.0

============================================================
Фільтр A: reinforcement learning, категорія cs.LG, після 2003
============================================================
#1 score=0.8297
Title: A neural network approach to ordinal regression
Category: cs.LG | Year: 2007.0

#2 score=0.7961
Title: A Novel Model of Working Set Selection for SMO Decomposition Methods
Category: cs.LG | Year: 2007.0

#3 score=0.7852
Title: Supervised Feature Selection via Dependence Estimation
Category: cs.LG | Year: 2007.0

#4 score=0.7851
Title: Statistical Mechanics of Nonlinear On-line Learning for Ensemble Teachers
Category: cs.LG | Year: 2007.0

#5 score=0.7776
Title: Parametric Learning and Monte Carlo Optimization
Category: cs.LG | Year: 2007.0

============================================================
Фільтр B: до 2007 року
============================================================
#1 score=0.8682
Title: Multi-Agent Modeling Using Intelligent Agents in the Game of Lerpa
Category: cs.MA | Year: 2007.0

#2 score=0.8324
Title: Modeling the field of laser welding melt pool by RBFNN
Category: physics.comp-ph | Year: 2007.0

#3 score=0.8297
Title: A neural network approach to ordinal regression
Category: cs.LG | Year: 2007.0

============================================================
Порівняння метрик схожості
============================================================
--- Cosine ---
#1 [cs.LG] A neural network approach to ordinal regression
#2 [math.ST] Multilayer Perceptron with Functional Inputs
#3 [physics.comp-ph] Modeling the field of laser welding melt pool by RBFNN
#4 [cs.AI] Comparing Robustness of Pairwise and Multiclass Neural-Network Systems
#5 [cs.CV] Automatic Detection of Pulmonary Embolism using Computational Intelligence

--- Dot Product ---
#1 [cs.LG] A neural network approach to ordinal regression
#2 [math.ST] Multilayer Perceptron with Functional Inputs
#3 [physics.comp-ph] Modeling the field of laser welding melt pool by RBFNN
#4 [cs.AI] Comparing Robustness of Pairwise and Multiclass Neural-Network Systems
#5 [cs.CV] Automatic Detection of Pulmonary Embolism using Computational Intelligence

--- L2 (негативна) ---
#1 [cs.LG] A neural network approach to ordinal regression
#2 [math.ST] Multilayer Perceptron with Functional Inputs
#3 [physics.comp-ph] Modeling the field of laser welding melt pool by RBFNN
#4 [cs.AI] Comparing Robustness of Pairwise and Multiclass Neural-Network Systems
#5 [cs.CV] Automatic Detection of Pulmonary Embolism using Computational Intelligence
```

### 05_chunking.py

```
Індекс arxiv-chunks-fixed створено
Індекс arxiv-chunks-semantic створено
Готуємо fixed chunks...
Fixed chunks: 236
100%|████████████████████████████████| 3/3 [00:09<00:00,  3.17s/it]
Готуємо semantic chunks...
Semantic chunks: 193
100%|████████████████████████████████| 2/2 [00:07<00:00,  4.00s/it]

Fixed chunks в індексі: 236
Semantic chunks в індексі: 193
```

### 06_hybrid_search.py

```
############################################################
ЗАПИТ: BERT fine-tuning
############################################################
BM25 #1: The NMSSM Solution to the Fine-Tuning Problem [hep-ph] score=11.50
Vector #1: Misere quotients for impartial games [math.CO] score=0.8645
Hybrid #1: The NMSSM Solution to the Fine-Tuning Problem [hep-ph] rrf=0.016393

############################################################
ЗАПИТ: Yann LeCun convolutional networks
############################################################
BM25 #1: On Punctured Pragmatic Space-Time Codes [cs.IT] score=13.48
Vector #1: Multilayer Perceptron with Functional Inputs [math.ST] score=0.8479
Hybrid #1: Optimization in Gradient Networks [cond-mat] rrf=0.030303

############################################################
ЗАПИТ: making computers understand human emotions from text
############################################################
BM25 #1: An Automated Evaluation Metric for Chinese Text Entry [cs.HC] score=18.27
Vector #1: Opinion Dynamics and Sociophysics [physics.soc-ph] score=0.8287
Hybrid #1: On the Development of Text Input Method [cs.CL] rrf=0.032258
```

---

## Частина 1 — Теоретичні питання

### 1. Чим Pinecone відрізняється від Qdrant і Chroma?

**Pinecone** — повністю керований хмарний сервіс (SaaS). Не потребує локального розгортання, масштабується автоматично. Безкоштовний тір дозволяє один проєкт і до 100 000 векторів. Ліцензія пропрієтарна — код закритий. Підходить коли потрібно швидко запустити продакшн без DevOps-витрат.

**Qdrant** — open-source, можна розгортати локально або в хмарі. Має власний хмарний тір. Підтримує складні фільтри, payload-індекси, квантизацію. Ліцензія Apache 2.0. Підходить для продакшн-систем де потрібен повний контроль над даними та інфраструктурою.

**Chroma** — легковаговий open-source, орієнтований на локальну розробку та прототипування. Найпростіший у налаштуванні, працює in-memory або на диску. Ліцензія Apache 2.0. Підходить для швидкого прототипування та невеликих датасетів.

Коли що обирати: Pinecone — продакшн без DevOps; Qdrant — продакшн з контролем над інфраструктурою; Chroma — локальна розробка і прототипи.

### 2. Чому обрана модель specter2_base, а не all-MiniLM-L6-v2?

`specter2_base` навчена спеціально на наукових текстах. Згідно з карткою моделі на HuggingFace, вона навчена для задач "paper search, recommendation, and classification" на корпусі наукових статей з семантичними зв'язками між цитуваннями. Модель розуміє специфічну термінологію фізики, математики, CS.

`all-MiniLM-L6-v2` — універсальна модель навчена на загальних текстах. Вона менша (22M параметрів проти 110M) і швидша, але не розуміє наукову термінологію так само добре. Для запиту "quantum chromodynamics energy levels" вона може не відрізнити релевантну фізичну статтю від нерелевантного тексту з подібними словами.

### 3. Рекомендована метрика схожості для specter2_base

Картка моделі рекомендує cosine similarity як метрику схожості. Це важливо при створенні індексу тому що Pinecone оптимізує пошук залежно від обраної метрики — якщо створити індекс з метрикою dotproduct або euclidean, а модель навчена на cosine, результати пошуку будуть менш точними навіть якщо математично обидва варіанти дають схоже ранжування для нормалізованих векторів.

### 4. Чому cosine similarity еквівалентна dot product для нормалізованих ембеддингів?

Cosine similarity визначається як `cos(a, b) = (a · b) / (||a|| * ||b||)`. Якщо вектори нормалізовані (||a|| = ||b|| = 1), то знаменник дорівнює 1, і формула спрощується до `cos(a, b) = a · b`. Тобто cosine similarity стає рівною dot product. Саме тому у результатах скрипту `04_search.py` топ-5 для Cosine і Dot Product повністю збіглися.

---

## Частина 3 — Теоретичні питання

### Чи збігаються топ-5 для cosine і dot product?

Так, збігаються повністю — всі 5 позицій однакові для обох метрик. Це очікуваний результат оскільки ембеддинги нормалізовані (норма = 1.0, що підтверджено у виводі `02_embed.py`). Математично cosine similarity для одиничних векторів тотожня dot product.

### Чи відрізняються результати для L2?

У нашому випадку топ-5 для L2 також збіглися з cosine і dot product. Це пояснюється тим що для нормалізованих векторів мінімальна L2-відстань відповідає максимальному cosine — тому ранжування збігається.

### Що сталося б якби ембеддинги не були нормалізовані?

Без нормалізації dot product залежав би від довжини векторів — довші вектори мали б вищий dot product навіть якщо вони семантично менш релевантні. Cosine нормалізує це автоматично. L2 давала б інші результати. Всі три метрики давали б різні топ-5.

---

## Частина 4 — Теоретичні питання

### Яка стратегія дає більш осмислені чанки?

Semantic chunking дає більш осмислені чанки оскільки зберігає цілі речення. Fixed chunking ріже текст механічно по кількості слів, через що речення можуть бути розрізані посередині. У результатах видно що semantic chunks містять завершені думки, тоді як fixed chunks іноді починаються або закінчуються на середині речення.

### Чи є випадки розрізаних речень?

Так, у fixed chunking є розрізані речення — це неминуче при механічному розбитті по кількості слів. Розрізане речення погіршує якість ембеддингу тому що модель отримує неповний контекст. Наприклад, чанк що починається з "of recursive (hierarchical) lattices" без попереднього контексту важко інтерпретувати правильно.

### Як overlap впливає на кількість чанків і покриття?

Більший overlap збільшує кількість чанків але покращує покриття контексту на межах — кожен чанк містить частину попереднього, що зменшує втрату інформації при розрізанні. У нашому випадку overlap=10 при chunk_size=50 дав 236 fixed chunks проти 193 semantic chunks.

---

## Частина 5 — Порівняльна таблиця гібридного пошуку

| Запит                                                  | BM25 #1                                                       | Vector #1                                              | Hybrid RRF #1                                          |
| ------------------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| "BERT fine-tuning"                                     | The NMSSM Solution to the Fine-Tuning Problem [hep-ph]        | Misere quotients for impartial games [math.CO]         | The NMSSM Solution to the Fine-Tuning Problem [hep-ph] |
| "Yann LeCun convolutional networks"                    | On Punctured Pragmatic Space-Time Codes [cs.IT]               | Multilayer Perceptron with Functional Inputs [math.ST] | Optimization in Gradient Networks [cond-mat]           |
| "making computers understand human emotions from text" | An Automated Evaluation Metric for Chinese Text Entry [cs.HC] | Opinion Dynamics and Sociophysics [physics.soc-ph]     | On the Development of Text Input Method [cs.CL]        |

### Який метод дав кращий результат?

Для точних термінів як "BERT fine-tuning" BM25 знаходить статті зі словом "fine-tuning" — хоча і в іншому контексті, це демонструє що BM25 чутливий до точних збігів токенів. Векторний пошук виграє на перефразуваннях — запит "making computers understand human emotions from text" знайшов статтю про opinion dynamics яка семантично близька але не містить жодного з цих слів дослівно. Гібридний RRF у третьому запиті вивів на перше місце статтю яка присутня і в BM25 (#2) і у vector (#2) — це класичний приклад переваги RRF.

### Чи є документи в гібридному топ-5 яких немає в окремих методах?

Так — у запиті "Yann LeCun convolutional networks" гібридний пошук вивів на #1 "Optimization in Gradient Networks" якої не було в топ-5 ні BM25 ні vector. Це відбувається тому що RRF підсумовує ранги — документ який займає середню позицію в обох методах отримує вищий RRF-скор ніж документ що займає #1 лише в одному методі.

### Як зміна параметра k в RRF впливає на видачу?

При k=60 вплив різниці між рангами згладжується — документ на #1 і документ на #10 мають близькі внески (1/61 vs 1/70). При k=1 різниця між рангами різкіша — #1 дає 1/2, #10 дає 1/11, тобто топові результати домінують значно сильніше. Малий k робить RRF більш агресивним і ближчим до winner-takes-all.

---

## Частина 6 — Аналіз і висновки

### 1. Семантичний пошук vs BM25

BM25 виграє на точних термінах і абревіатурах. Запит "BERT fine-tuning" через BM25 знайшов статті зі словом "fine-tuning" — хоча і в іншому контексті. Векторний пошук виграє на перефразуваннях — запит "making computers understand human emotions from text" знайшов статтю про opinion dynamics яка семантично близька але не містить жодного з цих слів дослівно. Загальне правило: BM25 для термінологічних і точних запитів, векторний для концептуальних і перефразованих.

### 2. Вплив розміру чанка

Занадто малий чанк (10-15 слів) втрачає контекст — окреме речення без оточення може бути неоднозначним і ембеддинг буде нестабільним. Занадто великий чанк (500+ слів) змішує кілька ідей в один вектор — пошук знайде чанк але він буде релевантним лише частково. Оптимальний розмір залежить від задачі: для наукових анотацій розмір 50-100 слів з overlap 10-20 дає хороший баланс між контекстом і точністю.

### 3. Невідповідна метрика

Якби індекс Pinecone був створений з метрикою euclidean але модель повертає нормалізовані вектори, математично результати були б еквівалентними cosine через те що для одиничних векторів мінімальна L2-відстань відповідає максимальному cosine. Проблема виникла б якби вектори не були нормалізовані — тоді L2 і cosine давали б різні результати, і невірна метрика в індексі призводила б до поганої якості пошуку.

### 4. Обмеження Pinecone Starter

Безкоштовний тір обмежує до одного проєкту і приблизно 100 000 векторів. У цьому завданні використано три індекси що наближається до ліміту. Для 10 мільйонів статей потрібен платний тір або альтернативне рішення: шардування індексу по категоріях, квантизація векторів для зменшення розміру, зберігання повних метаданих окремо у PostgreSQL або MongoDB з підтягуванням за ID після пошуку у Pinecone.
