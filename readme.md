# 🕵️ AI Dialogue Generator for Fraud Detection Research

## 📖 Overview

This project provides a comprehensive framework for generating and processing dialogue datasets specifically designed for fraud detection AI training. The system creates both fraudulent (positive) and non-fraudulent (negative) dialogue examples using multiple AI models.

## 🚀 Quick Start

### Dialog Generation

**Fraudulent Dialogues (Positive Examples)**

```
# Configure API settings in config/settings/
python main_with_background.py
```

### Non-Fraudulent Dialogues (Negative Examples)

```
# Configure prompts in config/settings/
python src/main.py
```

## 📊 Modules

### 🔍 Dialog | Fraudulent Dialogue Generation

**Purpose**: Generate realistic fraudulent conversations for training detection models

**Features**:

- 

  🤖 **Multi-Model Support**: Utilizes DeepSeek, DouBao, and Kimi AI models

- 

  📰 **Real-based Generation**: Trained on actual scam cases from Fraud_News_Reports dataset

- 

  ⚙️ **Configurable**: Easy API and dataset path configuration

**Usage**:

1. 

   Edit model configurations in `config/settings/`

2. 

   Run generation script

3. 

   Output: Authentic fraudulent dialogue dataset

### 🛡️ NotFraudDialog | Benign Dialogue Generation

**Purpose**: Create non-fraudulent conversations to balance training data

**Features**:

- 

  🎭 **Scenario-based**: Uses predefined scenarios from `src/core/dialogue_generator.py`

- 

  ⚖️ **Balanced Data**: Ensures diverse non-fraudulent examples

- 

  🔧 **Customizable**: Easy prompt engineering for different scenarios

### 💾 DataHandle | Data Processing Pipeline

#### 🔄 Processing Flow

```
Raw Data → Merge → Deduplicate → Anonymize → Filter → Split → Balance → Final Dataset
```

#### 📁 Scripts Overview

**`main2.py`** - Preliminary Processing

- 

  **Input**: Raw generated dialogues

- 

  **Processing**: Merging + Deduplication + Anonymization

- 

  **Output**: Analysis-ready data for threshold determination

**`main.py`** - Full Pipeline

- 

  **Input**: Preliminary processed data

- 

  **Processing**: Length filtering + Dataset splitting + Class balancing

- 

  **Output**: Final training-ready dataset

#### 🛠️ Tools

**`csvDistribution`** - Statistical Analysis

- 

  📈 Generates distribution plots and statistics

- 

  📏 Calculates median lengths for dialogue segmentation

- 

  🎯 Determines optimal length thresholds

**`DeleteNum`** - Threshold Optimization

- 

  📊 Analyzes anonymized data characteristics

- 

  ⚡ Identifies optimal filtering parameters

- 

  🔍 Ensures data quality standards

## 🏋️ Training & Evaluation

### Model Training

```
# Configure paths in config/config.py
torchrun --nproc_per_node=x main.py
```

### Evaluation

**`test.py`** - Model Performance Assessment

- 

  🧪 Evaluate base model (leave CheckPoint empty)

- 

  🔗 Evaluate fine-tuned model (specify CheckPoint path)

- 

  📊 Comprehensive performance metrics

## ⚙️ Configuration

### Settings Structure

```
config/
└── settings/
    ├── model_apis.yaml      # AI model configurations
    ├── dataset_paths.yaml   # Data source paths
    └── processing_params.yaml # Processing parameters
```

## 📈 Output

- 

  🗂️ Balanced dialogue datasets (fraud/benign)

- 

  📊 Statistical analysis reports

- 

  🎯 Optimized training-ready data splits

- 

  📝 Comprehensive documentation

------

*This project supports academic research in fraud detection and AI security. Use responsibly.*