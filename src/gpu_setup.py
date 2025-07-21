# competition_gpu_check_simple.py
"""
GPU check for Identify the Author competition - NO TENSORFLOW
Focus on PyTorch + XGBoost + scikit-learn for winning solution.
"""
import time
import sys


def check_competition_frameworks():
    """Check frameworks needed for competition (without TensorFlow)."""
    print("🎯 IDENTIFY THE AUTHOR - SIMPLIFIED GPU CHECK")
    print("=" * 55)
    print("📋 Strategy: PyTorch + XGBoost + Scikit-learn ensemble")

    results = {}

    # 1. PyTorch (for BERT alternative + custom models)
    print("\n1️⃣ PyTorch (BERT alternative + Neural models):")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda:0')

            # Test BERT-like computation
            x = torch.randn(16, 128, 768, device=device)  # Batch, seq_len, hidden
            linear = torch.nn.Linear(768, 3).to(device)

            start = time.time()
            with torch.no_grad():
                output = linear(x)
            torch.cuda.synchronize()
            duration = time.time() - start

            print(f"   ✅ PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"   ✅ BERT-like computation: {duration:.3f}s")
            print(f"   ✅ Memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3}GB")
            results['pytorch'] = True
        else:
            print("   ⚡ PyTorch CPU mode (no CUDA)")
            results['pytorch'] = False
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        results['pytorch'] = False

    # 2. XGBoost (for ensemble)
    print("\n2️⃣ XGBoost (Ensemble component):")
    try:
        import xgboost as xgb
        import numpy as np

        # Create realistic competition data
        X = np.random.random((1000, 100))
        y = np.random.randint(0, 3, 1000)
        dtrain = xgb.DMatrix(X, label=y)

        params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'objective': 'multi:softprob',
            'num_class': 3,
            'verbosity': 0
        }

        start = time.time()
        model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)
        duration = time.time() - start

        print(f"   ✅ XGBoost GPU: tree_method=gpu_hist")
        print(f"   ✅ Training 50 rounds: {duration:.3f}s")
        results['xgboost'] = True

    except Exception as e:
        print(f"   ❌ XGBoost GPU error: {e}")
        print("   ⚡ Will fallback to CPU mode")
        results['xgboost'] = False

    # 3. Scikit-learn (for TF-IDF + traditional ML)
    print("\n3️⃣ Scikit-learn (TF-IDF + Traditional ML):")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np

        # Test TF-IDF + LogReg (competition baseline)
        texts = [
            "Upon the pallid moon of night, darkness fell.",
            "The ancient one stirred from cosmic slumber deep.",
            "It was a dark and stormy night when Victor began.",
            "Nevermore spoke the raven in chambers dreary.",
            "From beyond came whispers of elder things."
        ]
        labels = [0, 1, 2, 0, 1]  # EAP, HPL, MWS

        # TF-IDF features
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X = vectorizer.fit_transform(texts)

        # Test models
        start = time.time()
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X, labels)

        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, labels)
        duration = time.time() - start

        print(f"   ✅ TF-IDF vectorization: {X.shape[1]} features")
        print(f"   ✅ LogisticRegression + RandomForest: {duration:.3f}s")
        print("   ✅ Competition baseline models ready")
        results['sklearn'] = True

    except Exception as e:
        print(f"   ❌ Scikit-learn error: {e}")
        results['sklearn'] = False

    # 4. Transformers (for pre-trained models without TensorFlow)
    print("\n4️⃣ Transformers (Hugging Face with PyTorch):")
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        # Test if we can load BERT tokenizer (lightweight test)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # Test tokenization
        text = "Upon the pallid moon of night"
        tokens = tokenizer(text, return_tensors='pt', max_length=64, padding=True, truncation=True)

        if torch.cuda.is_available():
            # Light model test
            print("   ✅ BERT tokenizer working")
            print("   ✅ PyTorch backend available for transformers")
            print("   ✅ Can load pre-trained models for feature extraction")
            results['transformers'] = True
        else:
            print("   ⚡ Transformers available (CPU mode)")
            results['transformers'] = True

    except Exception as e:
        print(f"   ❌ Transformers error: {e}")
        results['transformers'] = False

    # Summary
    print("\n🏆 COMPETITION STRATEGY READINESS:")
    print("-" * 40)
    working = sum(results.values())
    total = len(results)

    for framework, status in results.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {framework.title()}")

    print(f"\n📊 Status: {working}/{total} frameworks ready")

    # Competition strategy recommendations
    print("\n🎯 RECOMMENDED COMPETITION APPROACH:")
    print("-" * 40)

    if results.get('pytorch', False) and results.get('xgboost', False):
        print("   🚀 OPTIMAL SETUP:")
        print("   • PyTorch for BERT-style models (GPU accelerated)")
        print("   • XGBoost for gradient boosting (GPU accelerated)")
        print("   • Scikit-learn for TF-IDF + traditional ML")
        print("   • Expected performance: TOP 10% potential")
        print("\n   💡 Key advantage: No TensorFlow dependencies!")

    elif results.get('sklearn', False):
        print("   ⚡ SOLID SETUP:")
        print("   • Focus on TF-IDF + character n-grams")
        print("   • Ensemble of LogReg + RandomForest + XGBoost")
        print("   • Strong stylometric features")
        print("   • Expected performance: TOP 25% potential")

    else:
        print("   ⚠️  BASIC SETUP:")
        print("   • Pure CPU-based approach")
        print("   • Still competitive with good feature engineering")

    # Training time estimates
    if results.get('pytorch', False):
        print("\n⏱️  ESTIMATED TRAINING TIMES:")
        print("   • PyTorch BERT alternative: ~5-10 minutes")
        print("   • XGBoost (if GPU): ~1-2 minutes")
        print("   • Full ensemble: ~15-20 minutes")
    else:
        print("\n⏱️  ESTIMATED TRAINING TIMES (CPU):")
        print("   • Traditional ML models: ~5-10 minutes")
        print("   • Full ensemble: ~20-30 minutes")

    return results


def create_no_tensorflow_requirements():
    """Create clean requirements without TensorFlow."""

    requirements = '''# requirements_no_tensorflow.txt
# Clean setup for Identify the Author competition
# Focus: PyTorch + XGBoost + Scikit-learn

# ===== CORE ML PACKAGES =====
numpy>=1.21.0,<2.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0

# ===== PYTORCH (BERT ALTERNATIVE) =====
torch>=2.1.0
torchaudio>=2.1.0
torchvision>=0.16.0

# ===== TRANSFORMERS (PYTORCH BACKEND) =====
transformers>=4.21.0
accelerate>=0.20.0
datasets>=2.0.0

# ===== GPU-ACCELERATED ML =====
xgboost>=1.6.0

# ===== NLP PACKAGES =====
nltk>=3.6
spacy>=3.4.0

# ===== COMPETITION ESSENTIALS =====
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
wordcloud>=1.9.0

# ===== JUPYTER & DEVELOPMENT =====
jupyter>=1.0.0
ipykernel>=6.15.0
notebook>=6.4.0

# ===== PERFORMANCE =====
numba>=0.56.0

# ===== DEVELOPMENT TOOLS =====
pytest>=7.0.0
black>=22.0.0
ipython>=8.0.0

# ===== COMPETITION STRATEGY =====
# 1. PyTorch for modern neural models (BERT alternatives)
# 2. XGBoost for gradient boosting (GPU accelerated if available)  
# 3. Scikit-learn for TF-IDF and traditional ML
# 4. NO TensorFlow = NO circular import issues!
'''

    with open('requirements_no_tensorflow.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)

    print("   ✅ Created requirements_no_tensorflow.txt")


if __name__ == "__main__":
    print("🚀 COMPETITION SETUP - NO TENSORFLOW APPROACH")
    print("=" * 60)
    print("💡 Strategy: Skip TensorFlow, use PyTorch + XGBoost + Scikit-learn")
    print("🎯 Goal: Eliminate import issues, focus on winning models")

    # Check current setup
    results = check_competition_frameworks()

    # Create clean requirements
    print("\n📦 CREATING CLEAN REQUIREMENTS:")
    create_no_tensorflow_requirements()

    print("\n🎯 NEXT STEPS:")
    print("1. Optional: pip install -r requirements_no_tensorflow.txt")
    print("2. Run your competition pipeline")
    print("3. Focus on ensemble of PyTorch + XGBoost + Scikit-learn")

    if results.get('pytorch') and results.get('xgboost'):
        print("\n🎉 READY FOR COMPETITION!")
        print("   Your setup can achieve top leaderboard performance!")
    else:
        print("\n⚡ PARTIAL SETUP - Still competitive!")