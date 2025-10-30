# IBM Attrition Prediction — ML Aplicado

Modelo de predicción de **attrition** (renuncia de empleados por desgaste) con el dataset sintético de IBM. Este repo contiene el desarrollo del proyecto, desde el EDA, la preparación de datos, pipelines de entrenamiento y resultados.

---

## 🗂️ Estructura del repositorio

```
ml-aplicado-attrition-prediction-IBM/
├─ ENTREGA 1/         # Notebooks y artefactos de la entrega (EDA, preparación, modelado)
├─ ENTREGA 2/         # Notebooks y artefactos de la entrega (FE, tuning y umbrales)
├─ .gitignore
└─ README.md
```

> Nota: Estructura observada en la rama `main`. Los notebooks de trabajo se encuentran en **ENTREGA XX/procesamiento**.

---

## 📦 Dataset - Contexto

La rotación de empleados es un desafío común en todas las empresas, ya que genera costos importantes relacionados con la interrupción de procesos, la contratación y la capacitación de nuevo personal. Para enfrentar este problema, los modelos de clasificación pueden ayudar a predecir qué empleados tienen mayor probabilidad de renunciar, lo que permite a Recursos Humanos intervenir a tiempo. Sin embargo, el éxito no depende solo del modelo, sino también del factor humano: hablar con el empleado, entender su situación y actuar sobre aspectos que se pueden controlar.

El dataset de IBM utilizado para este análisis es limitado en tamaño, lo que implica que los modelos solo ofrecen una mejora moderada frente al azar. Aun así, entender y reducir la rotación, además de prepararse para los casos inevitables, puede mejorar notablemente las operaciones de una organización. Con un conjunto de datos más grande, sería posible segmentar empleados en categorías de riesgo y obtener insights más profundos sobre las causas de la rotación, generando aprendizajes más valiosos que los que se logran solo con entrevistas.

## 📦 Dataset — Data Card

**Nombre:** IBM HR Analytics – Employee Attrition & Performance  
**Origen:** conjunto **ficticio/sintético** creado por IBM para análisis de rotación.  
**Tarea:** **clasificación binaria** (`Attrition`: Yes/No).  
**Usos típicos:** detección de riesgo de renuncia, priorización de acciones de *retención*, análisis de factores asociados.

**Ubicación:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data


### Composición
- **Tamaño:** **1,470** filas × **35** columnas.  
- **Tipos de variables:**
  - **Numéricas** (p. ej., `Age`, `MonthlyIncome`, `DistanceFromHome`).
  - **Categóricas nominales** (`Department`, `JobRole`, `Gender`, `BusinessTravel`, `MaritalStatus`).
  - **Categóricas ordinales** codificadas como enteros: `Education` (1–5), `Job/Environment/RelationshipSatisfaction` (1–4), `JobInvolvement` (1–4), `WorkLifeBalance` (1–4), `PerformanceRating` (1–4), `StockOptionLevel` (0–3).
- **Columnas constantes / poca utilidad:** `EmployeeCount`, `StandardHours` (constantes) y `Over18` (casi siempre “Yes”).
- **Identificador:** `EmployeeNumber` (ID único) → **no** usar como predictor (solo para trazabilidad).

### Calidad de datos
- **Nulos:** dataset consultado desde Kaggle **sin valores faltantes**.  
- **Desbalance:** la clase `Attrition=Yes` es minoritaria (desbalance moderado del 16%).  
- **Ordinalidad:** tratar escalas 1–4/1–5 como **ordinales** cuando el modelo lo permita.


## ⚙️ Cómo ejecutar

![⚠️ IMPORTANTE](https://img.shields.io/badge/%E2%9A%A0%EF%B8%8F-WARNING-red?style=for-the-badge) 

Para una correcta ejecución en Colab, se debe usar el runtime 2025.07 que habilita dependencias internas requeridas por la librería `pycaret`

1. El notebook fue ejecutado en Colab, se sugiere validar su ejecución también en Colab
2. Cargar directamente el notebook, el archivo de scripts utils.py con métodos para aplicar EDA y el dataset local dataset_ibm.csv. Si se desea ejecutar solo el tuning, se debe cargar el dataset local dataset_ibm_sin_outliers.csv, resultante de la primera etapa de modelación.
3. Entrega 1: Ejecutar el flujo: **EDA → preparación → baseline** (Opcional si se remite directamente al tuning)
4. Entrega 2: Ejecutar el flujo: **FE → pipelines → tuning → análisis de umbrales**

---

## 📈 Resultados preliminares

### Entrega 1:

**Modelo baseline:** Regresión Logística  
**Métricas(conjunto de test):** Accuracy **89.4%** · AUC **90.7%** · Recall **57.9%** · Precision **64.7%** · F1 **61.1%** · Kappa **55.5%** 

**Interpretación preliminar:** buen equilibrio general, con margen para aumentar **recall** de la clase positiva mediante **ajuste de umbral**, `class_weight` o **calibración**, y oportunidades de mejora mediante la implementación de **ingeniería de características**

### Entrega 2:
**Modelo candidato:** XGBoost  
**Métricas(conjunto de test):** Accuracy **80.3%** · AUC **88.6%** · Recall **76.1%** · Precision **68.4%** · F1 **70.6%** · Kappa **42.1%** 

**Interpretación preliminar:** se mantiene el equilibrio general, evidenciandose mejoría en el recall y f1 score, generando mayor confianza en el modelo para apoyar la retención de empleados

---

## 🚀 Próximos pasos

Entrega 1 (HECHO):
- Ajuste de **umbral** y **calibración** (Platt/Isotónica) para mejorar recall manteniendo precisión.  
- Generar **Interacciones/transformaciones** útiles en la fase de ingeniería de características
- Comparar con **boosting** (LightGBM/XGBoost) manteniendo la logística como referencia.  
- Reporte de métricas por grupos poblacionales y **entrenamiento robusto**

Entrega 2:
- Calibración de modelo candidato
- Importancia de características
- Conclusiones y despliegue

---

## 🤝 Créditos

Este proyecto está siendo desarrollado por:
- ### Andrea Paola Alzate Ramirez
- ### Gustavo Adolfo Jerez Tous
