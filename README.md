# AI on the Edge — TinyML for Embedded Vaccine Cold-Chain Monitoring

> **Bachelor's thesis project — CTJ / Federal University of Santa Catarina (UFSC)**
> Investigating the deployment of artificial intelligence models on ESP32 microcontrollers for thermal monitoring of vaccines.

---

## Overview

This project explores the feasibility of running machine learning models directly on resource-constrained embedded hardware. Models originally trained in a full computational environment were adapted and ported to the **ESP32 microcontroller** using TinyML techniques, demonstrating that effective AI inference is achievable even under strict memory and processing constraints.

The primary application domain is **vaccine cold-chain monitoring** — a safety-critical context where low-cost, autonomous, and intelligent edge devices can make a real-world difference.

---

## Motivation

Microcontrollers are ubiquitous in IoT deployments, but their limited resources (RAM, flash, and no floating-point unit in many variants) make running conventional ML models impractical. This work bridges the gap between:

- Models developed by students at **CTJ-UFSC** in standard Python/TensorFlow environments
- Deployment on **ESP32** via TensorFlow Lite and the `emlearn` C-header approach

Despite a slight performance reduction compared to the original models, results confirmed **satisfactory accuracy** on-device — validating the viability of TinyML in embedded, real-world scenarios.

---

## Project Structure

```
.
├── data/
│   ├── raw/                  ← Raw Arduino serial captures (.txt)
│   └── processed/            ← Normalized and concatenated datasets (.xlsx)
│
├── models/
│   ├── SRNN/                 ← Simple RNN variants (.keras + training plots)
│   ├── GRU/                  ← GRU variants (.keras + training plots)
│   ├── LSTM/                 ← LSTM variants (.keras + training plots)
│   └── tflite/               ← TensorFlow Lite converted models
│       ├── SRNN/
│       ├── GRU/
│       └── LSTM/
│
├── embedded/
│   ├── decision_tree/        ← C headers generated via emlearn & micromlgen
│   └── arduino/              ← Arduino sketch for ESP32
│
├── results/
│   ├── metrics/              ← Classification reports, MSE/MAE logs
│   └── plots/                ← Prediction comparison plots
│
├── src/
│   ├── preprocess.py         ← Data loading & preparation (regression + classification)
│   ├── models.py             ← All 15 RNN model definitions (SRNN / GRU / LSTM)
│   ├── train.py              ← Full training pipeline with CLI
│   ├── evaluate.py           ← Evaluation for .keras and .tflite models
│   ├── convert.py            ← Keras → TFLite and Decision Tree → C header
│   └── serial_read.py        ← Arduino serial port data capture
│
├── config.yaml               ← Centralized project parameters
├── requirements.txt
└── README.md
```

---

## Methodology

The pipeline follows four stages:

```
Raw sensor data  →  Model training  →  Model conversion  →  ESP32 deployment
(Arduino serial)    (Keras / sklearn)   (TFLite / emlearn)   (TinyML inference)
```

**Three model families** were evaluated, each in five architectural variants:

| # | Model              | Family |
|---|--------------------|--------|
| 1 | SRN (baseline)     | SRNN   |
| 2 | SRNN-Deep          | SRNN   |
| 3 | SRNN-Dropout       | SRNN   |
| 4 | SRNN-Bidirectional | SRNN   |
| 5 | SRNN-Large         | SRNN   |
| 6 | GRU (baseline)     | GRU    |
| 7 | GRU-Deep           | GRU    |
| 8 | GRU-Dropout        | GRU    |
| 9 | GRU-Bidirectional  | GRU    |
|10 | GRU-Large          | GRU    |
|11 | LSTM (baseline)    | LSTM   |
|12 | LSTM-Deep          | LSTM   |
|13 | LSTM-Dropout       | LSTM   |
|14 | LSTM-Bidirectional | LSTM   |
|15 | LSTM-Large         | LSTM   |

**Decision trees** were also explored as a lightweight alternative, exported directly as C headers for integration into Arduino sketches via `emlearn` and `micromlgen`.


## Key Results

- All 15 model variants were successfully converted to `.tflite` and executed on ESP32
- Decision trees exported as C headers ran inference with **no external libraries** on the microcontroller
- A **slight accuracy reduction** was observed post-conversion, within acceptable margins for thermal monitoring
- Results validate **TinyML as a viable path** for IoT edge intelligence in safety-critical applications

---

## Embedded Libraries

| Library | Purpose |
|---------|---------|
| [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) | Running `.tflite` models on ESP32 |
| [emlearn](https://emlearn.org/) | Converting sklearn decision trees to C code |
| [micromlgen](https://github.com/eloquentarduino/micromlgen) | Alternative C header generation for Arduino |

---

## Application Context

Vaccine cold-chain integrity is a critical public health concern. Temperature excursions during storage or transport can silently compromise vaccine efficacy. This project demonstrates that a **low-cost ESP32 + ML sensor node** can perform intelligent thermal anomaly detection autonomously — without cloud connectivity — making it suitable for remote clinics, mobile vaccination units, and resource-limited environments.

---

## Academic Context

| | |
|---|---|
| **Institution** | CTJ — Federal University of Santa Catarina (UFSC) |
| **Type** | Bachelor's Thesis (TCC) |
| **Focus areas** | TinyML · Embedded Systems · IoT · Vaccine Cold Chain · Edge AI |
