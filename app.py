import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
import joblib

# Set Keras backend to tensorflow before importing Keras
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# Suppress Keras/TF logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Reconfigure stdout for Windows terminal compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# App title and page configuration
st.set_page_config(
    page_title="Dermatological Diagnostic Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clinical Taxonomy
DIAGNOSIS_DICT = {
    'nv': 'Melanocytic Nevus (Benign Mole)',
    'mel': 'Melanoma (Malignant Skin Cancer)',
    'bkl': 'Benign Keratosis-like Lesion',
    'bcc': 'Basal Cell Carcinoma (Malignant)',
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma (Pre-cancerous)',
    'vasc': 'Vascular Lesion (Benign)',
    'df': 'Dermatofibroma (Benign)'
}

MALIGNANCY_RISK = {
    'nv': 'Benign', 
    'mel': 'Malignant (Critical)', 
    'bkl': 'Benign',
    'bcc': 'Malignant', 
    'akiec': 'Pre-cancerous / In-situ',
    'vasc': 'Benign', 
    'df': 'Benign'
}

# Unified 3-Tier Risk Hierarchy
# High Risk (Red): mel, bcc, akiec
# Moderate Risk (Amber): Borderline / Low-confidence cases
# Low Risk (Green): nv, bkl, vasc, df
MALIGNANT_CLASSES = ["mel", "bcc", "akiec"]

RISK_TIERS = {
    "mel": {"tier": "High Risk (Malignant)", "color": "#D32F2F", "badge": "🔴 Malignant"},
    "bcc": {"tier": "High Risk (Malignant)", "color": "#D32F2F", "badge": "🔴 Malignant"},
    "akiec": {"tier": "High Risk (Pre-cancerous)", "color": "#D32F2F", "badge": "🔴 Pre-cancerous"},
    "nv": {"tier": "Low Risk (Benign)", "color": "#388E3C", "badge": "🟢 Benign"},
    "bkl": {"tier": "Low Risk (Benign)", "color": "#388E3C", "badge": "🟢 Benign"},
    "vasc": {"tier": "Low Risk (Benign)", "color": "#388E3C", "badge": "🟢 Benign"},
    "df": {"tier": "Low Risk (Benign)", "color": "#388E3C", "badge": "🟢 Benign"}
}

# Cache model loading to prevent reload overhead
@st.cache_resource
def load_assets():
    preprocessor = joblib.load('models/metadata_preprocessor.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    temp_dict = joblib.load('models/calibration_temperature.pkl')
    T = float(temp_dict['temperature'])
    
    # Extract fitted categorical dropdown options programmatically
    cat_encoder = preprocessor.named_transformers_['cat']
    sex_categories = list(cat_encoder.categories_[0])
    loc_categories = list(cat_encoder.categories_[1])
    
    # Load trained CNN model & feature extractor
    cnn_model = keras.models.load_model('models/best_cnn_model.keras')
    feature_extractor = keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("global_average_pooling2d").output
    )
    
    # Load fusion model & rebuild linear logit model
    fusion_model_path = 'models/best_fusion_model.keras'
    if not os.path.exists(fusion_model_path):
        raise FileNotFoundError(
            f"Required model file '{fusion_model_path}' not found! "
            "Please ensure 'best_fusion_model.keras' is located in the 'models/' directory."
        )
    fusion_model = keras.models.load_model(fusion_model_path)
    original_final_layer = fusion_model.layers[-1]
    penultimate_output = fusion_model.layers[-2].output
    
    logits_layer = keras.layers.Dense(7, activation=None, name="logits_output_app")
    logits_tensor = logits_layer(penultimate_output)
    logits_model = keras.Model(inputs=fusion_model.input, outputs=logits_tensor)
    logits_layer.set_weights(original_final_layer.get_weights())
    
    # Sanity check logit reconstitution
    dummy_input = np.random.rand(1, 1299)
    pred_orig = fusion_model(dummy_input, training=False).numpy()
    logits = logits_model(dummy_input, training=False).numpy()
    z = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    pred_reconstructed = exp_z / exp_z.sum(axis=1, keepdims=True)
    max_diff = np.abs(pred_orig - pred_reconstructed).max()
    sanity_status = f"PASSED (Max Diff: {max_diff:.8f})" if max_diff < 1e-4 else f"FAILED (Max Diff: {max_diff:.8f})"
    
    return preprocessor, label_encoder, T, feature_extractor, fusion_model, logits_model, sanity_status, sex_categories, loc_categories

# Load cached assets
with st.spinner("Initializing multimodal pipelines and models..."):
    preprocessor, label_encoder, T, feature_extractor, fusion_model, logits_model, sanity_status, sex_categories, loc_categories = load_assets()

# Sidebar - Patient Demographics & Image Input
st.sidebar.header("📋 Patient Demographics & Image Input")

# Quick Demo Case Button / Selector
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Benchmark Case Loader")
load_demo = st.sidebar.checkbox("Load Case 2 (`ISIC_0024700`) — Verified Slipped Case")

# Set default values based on demo loader
if load_demo:
    default_age = 35
    default_sex_idx = sex_categories.index('female') if 'female' in sex_categories else 0
    default_loc_idx = loc_categories.index('trunk') if 'trunk' in loc_categories else 0
else:
    default_age = 50
    default_sex_idx = 0
    default_loc_idx = 0

# Image file uploader
uploaded_file = st.sidebar.file_uploader("Upload Lesion Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Patient Age slider
age = st.sidebar.slider("Patient Age (years)", min_value=0, max_value=100, value=default_age, step=1)

# Programmatic Sex Selectbox
sex_display_options = [s.title() for s in sex_categories]
sex_selected = st.sidebar.selectbox("Patient Sex", sex_display_options, index=default_sex_idx)
selected_sex_raw = sex_categories[sex_display_options.index(sex_selected)]

# Programmatic Localization Selectbox
loc_display_options = [l.title() for l in loc_categories]
loc_selected = st.sidebar.selectbox("Lesion Site (Localization)", loc_display_options, index=default_loc_idx)
selected_loc_raw = loc_categories[loc_display_options.index(loc_selected)]

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Ethical & Clinical Disclaimer:** "
    "This application is a research prototype for decision support. "
    "It is not a certified medical device and does not replace expert dermatological examination."
)

# Header Title Banner
st.title("🔬 Automated Skin Lesion & Dermatological Disease Classification")
st.markdown("### *Multimodal Deep Learning Pipeline with Calibration & Asymmetric Safety Guard*")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Diagnostic Studio", 
    "📚 Skin Lesion Library", 
    "🛠️ Model Diagnostics & Calibration"
])

# ==============================================================================
# TAB 1: DIAGNOSTIC STUDIO
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    image_to_process = None
    image_bytes = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
        image_bytes = uploaded_file.getvalue()
    elif load_demo:
        demo_path = os.path.join("test_images", "ISIC_0024700.jpg")
        if os.path.exists(demo_path):
            image_to_process = Image.open(demo_path)
            with open(demo_path, "rb") as f:
                image_bytes = f.read()
    
    if image_to_process is not None and image_bytes is not None:
        with col1:
            st.subheader("🖼️ Uploaded Lesion Preview")
            st.image(image_to_process, use_container_width=True)
            st.info(f"**Metadata Summary:** Age: `{age}` | Sex: `{selected_sex_raw.title()}` | Site: `{selected_loc_raw.title()}`")
            
        with col2:
            st.subheader("🩺 Diagnostic Analysis & Safety Triage")
            
            with st.spinner("Computing multimodal inference..."):
                # A. TensorFlow Image Preprocessing
                img_tf = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
                shape = tf.shape(img_tf)
                h, w = shape[0], shape[1]
                
                # Downsample to 600x450 if non-standard resolution
                if h.numpy() != 450 or w.numpy() != 600:
                    img_tf = tf.image.resize(img_tf, [450, 600])
                    
                img_224 = tf.image.resize(img_tf, [224, 224])
                img_final = tf.cast(img_224, tf.float32)
                img_batch = tf.expand_dims(img_final, axis=0) # Shape: (1, 224, 224, 3)
                
                # Extract CNN features
                image_features = feature_extractor(img_batch, training=False).numpy()
                
                # B. Metadata Preprocessing
                meta_data = pd.DataFrame([{
                    'age': float(age),
                    'sex': selected_sex_raw,
                    'localization': selected_loc_raw
                }])
                meta_features = preprocessor.transform(meta_data)
                meta_features = np.asarray(meta_features.todense() if hasattr(meta_features, 'todense') else meta_features)
                
                # C. Feature Fusion
                fused_vector = np.concatenate([image_features, meta_features], axis=1) # Shape: (1, 1299)
                
                # D. Logits & Temperature Scaling Calibration
                raw_logits = logits_model(fused_vector, training=False).numpy()
                calibrated_logits = raw_logits / T
                
                z = calibrated_logits - calibrated_logits.max(axis=1, keepdims=True)
                exp_z = np.exp(z)
                calibrated_probs = (exp_z / exp_z.sum(axis=1, keepdims=True))[0]
                
                # E. Safety Decision Logic
                predicted_class_idx = calibrated_probs.argmax()
                predicted_class_code = label_encoder.classes_[predicted_class_idx]
                confidence = calibrated_probs[predicted_class_idx]
                
                malignant_idxs = [list(label_encoder.classes_).index(c) for c in MALIGNANT_CLASSES]
                malignant_mass = sum(calibrated_probs[i] for i in malignant_idxs)
                
                # Asymmetric Safety Rules
                if malignant_mass >= 0.15:
                    triage_decision = "FLAG FOR REVIEW (malignancy risk present)"
                    triage_status = "red"
                    triage_desc = (
                        f"The model detected significant probability mass on malignant/pre-cancerous classes "
                        f"({malignant_mass:.1%} total malignant risk mass $\\ge 15.0\\%$). Immediate specialist review is required."
                    )
                elif confidence >= 0.8:
                    triage_decision = "Model confident — likely benign"
                    triage_status = "green"
                    triage_desc = (
                        f"The model is confident ({confidence:.1%}) that this is a benign lesion. "
                        f"Malignant probability mass is low ({malignant_mass:.1%})."
                    )
                else:
                    triage_decision = "FLAG FOR REVIEW (low confidence)"
                    triage_status = "orange"
                    triage_desc = (
                        f"The model predicted a benign class but overall confidence is low ({confidence:.1%} < 80.0%). "
                        f"Manual dermatological verification is recommended."
                    )
                
                # Triage Banner Display
                if triage_status == "red":
                    st.error(f"🔴 **Triage Decision: {triage_decision}**\n\n{triage_desc}")
                elif triage_status == "orange":
                    st.warning(f"🟡 **Triage Decision: {triage_decision}**\n\n{triage_desc}")
                else:
                    st.success(f"🟢 **Triage Decision: {triage_decision}**\n\n{triage_desc}")
                
                # Primary Prediction Summary
                st.markdown(f"**Top Predicted Class:** `{predicted_class_code}` — **{DIAGNOSIS_DICT[predicted_class_code]}**")
                st.markdown(f"**Model Confidence:** `{confidence:.2%}` | **Malignant Risk Mass:** `{malignant_mass:.2%}`")
                
                st.markdown("---")
                st.markdown("#### 📊 Calibrated Probability Distribution (3-Tier Risk Hierarchy)")
                
                # Probability dataframe sorted by probability
                prob_df = pd.DataFrame({
                    'Class': [label_encoder.classes_[i] for i in range(len(calibrated_probs))],
                    'Diagnosis': [DIAGNOSIS_DICT[label_encoder.classes_[i]] for i in range(len(calibrated_probs))],
                    'Calibrated Probability': calibrated_probs,
                    'Tier': [RISK_TIERS[label_encoder.classes_[i]]['badge'] for i in range(len(calibrated_probs))]
                }).sort_values(by='Calibrated Probability', ascending=False)
                
                # Display progress bars styled by 3-tier risk system
                for _, row in prob_df.iterrows():
                    cls_code = row['Class']
                    tier_info = RISK_TIERS[cls_code]
                    prob_val = row['Calibrated Probability']
                    
                    st.write(f"{tier_info['badge']} **{cls_code.upper()}** - {row['Diagnosis']}")
                    st.progress(float(prob_val), text=f"{prob_val:.2%}")
    else:
        st.info("👈 Please upload a skin lesion image or select **Load Case 2** in the sidebar to run the analysis.")

# ==============================================================================
# TAB 2: SKIN LESION LIBRARY
# ==============================================================================
with tab2:
    st.subheader("📚 Clinical Taxonomy & Risk Classification Library")
    st.markdown(
        "This library defines the 7 dermatological classes from the HAM10000 dataset, "
        "structured under a unified **3-Tier Risk Hierarchy** for safety triage:"
    )
    
    # Tier 1: High Risk (Malignant & Pre-Cancerous)
    st.markdown("### 🔴 Tier 1: High Risk — Malignant & Pre-Cancerous Lesions")
    st.markdown(
        "These conditions require urgent dermatological evaluation and biopsy. "
        "Any probability mass assigned to these classes contributes directly to the primary $\\ge 15.0\\%$ safety triage threshold."
    )
    
    col_t1_a, col_t1_b, col_t1_c = st.columns(3)
    with col_t1_a:
        st.error("#### 🔴 MEL — Melanoma")
        st.markdown("**Malignant Skin Cancer**")
        st.caption(
            "Aggressive skin cancer arising from melanocytes. Requires immediate surgical excision and staging."
        )
    with col_t1_b:
        st.error("#### 🔴 BCC — Basal Cell Carcinoma")
        st.markdown("**Malignant Skin Cancer**")
        st.caption(
            "Common non-melanoma skin cancer originating from basal cells. Locally invasive with low metastasis risk."
        )
    with col_t1_c:
        st.error("#### 🔴 AKIEC — Actinic Keratosis / In-Situ")
        st.markdown("**Pre-Cancerous / Intraepithelial**")
        st.caption(
            "Dysplastic keratinocyte lesion induced by UV radiation. High risk of malignant transformation into invasive SCC; included in primary malignant triage mass."
        )
        
    st.markdown("---")
    
    # Tier 2: Moderate Risk / Clinical Watch
    st.markdown("### 🟡 Tier 2: Moderate Risk — Review Required")
    st.markdown(
        "Lesions predicted as benign but where the model's calibrated confidence is below **$80.0\\%$**. "
        "Manual specialist review is recommended due to classification uncertainty."
    )
    
    st.markdown("---")
    
    # Tier 3: Low Risk (Benign Lesions)
    st.markdown("### 🟢 Tier 3: Low Risk — Benign Conditions")
    st.markdown(
        "Non-cancerous skin conditions. Triaged as safe only when model confidence is $\\ge 80.0\\%$ and total malignant mass is $< 15.0\\%$."
    )
    
    col_t3_a, col_t3_b, col_t3_c, col_t3_d = st.columns(4)
    with col_t3_a:
        st.success("#### 🟢 NV — Melanocytic Nevus")
        st.markdown("**Benign Mole**")
        st.caption("Common benign proliferation of melanocytes.")
    with col_t3_b:
        st.success("#### 🟢 BKL — Benign Keratosis")
        st.markdown("**Benign Keratosis-like Lesion**")
        st.caption("Includes seborrheic keratoses, solar lentigines, and lichen planus-like keratoses.")
    with col_t3_c:
        st.success("#### 🟢 VASC — Vascular Lesion")
        st.markdown("**Benign Vascular Anomaly**")
        st.caption("Cherry angiomas, angiokeratomas, and pyogenic granulomas.")
    with col_t3_d:
        st.success("#### 🟢 DF — Dermatofibroma")
        st.markdown("**Benign Fibrous Nodule**")
        st.caption("Benign dermal nodule common on extremities.")

# ==============================================================================
# TAB 3: MODEL DIAGNOSTICS & CALIBRATION
# ==============================================================================
with tab3:
    st.subheader("🛠️ Model Performance, Calibration & Diagnostic Benchmarks")
    st.markdown(
        "Quantitative performance evaluation and calibration analysis for the "
        "Multimodal Joint Fusion Model (EfficientNetB0 + Patient Metadata)."
    )
    
    # Section A: Temperature Scaling & Logits Summary
    st.markdown("#### ⚙️ Temperature Scaling & Logit Extraction")
    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.info(f"**Optimal Calibration Temperature ($T$):** `{T:.6f}`")
    with col_diag2:
        st.info(f"**Logit Reconstitution Check:** `{sanity_status}`")
        
    st.markdown("---")
    
    # Section B: Quantitative Benchmark Tables
    col_tab_a, col_tab_b = st.columns(2)
    
    with col_tab_a:
        st.markdown("#### 📈 Overall Validation Performance Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Balanced Accuracy", "Macro Recall", "Macro Precision", "Macro F1-Score"],
            "Fusion Model": ["72.59%", "71.78%", "71.78%", "56.63%", "62.04%"]
        })
        st.table(metrics_df)
        
    with col_tab_b:
        st.markdown("#### 📊 Confidence-Threshold Trade-off Table")
        tradeoff_data = pd.DataFrame({
            "Threshold": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
            "Coverage": ["100.0%", "80.9%", "68.0%", "57.0%", "45.6%", "36.5%"],
            "Retained Accuracy": ["72.59%", "81.17%", "86.59%", "91.60%", "95.04%", "97.26%"],
            "Retained Recall": ["71.78%", "78.77%", "82.20%", "85.66%", "85.04%", "78.07%"]
        })
        st.table(tradeoff_data)
        
    st.markdown("---")
    
    # Section C: PRIMARY EVALUATION COMPARISON GALLERY (Agreed 6 Key Narrative Charts)
    st.markdown("### 🏆 Primary Evaluation Comparison Gallery (Model Narrative)")
    st.markdown(
        "These six core visual charts articulate the primary findings of the study: "
        "multimodal architecture comparison, per-class sensitivity gains, probability calibration, and safety trade-offs."
    )
    
    plots_dir = os.path.join("artifacts", "plots")
    
    primary_plots = [
        ("final_comparison_overall.png", "1. Overall Performance Across Architectures (RF vs CNN vs Fusion)"),
        ("final_comparison_per_class_recall.png", "2. Per-Class Sensitivity (Recall) — Where Multimodal Fusion Helps Most"),
        ("reliability_diagram_calibrated.png", "3. Calibration Reliability Diagram — Before vs. After Temperature Scaling"),
        ("confidence_threshold_tradeoff.png", "4. Confidence Thresholding — Coverage vs. Reliability Trade-off"),
        ("confusion_matrix_fusion.png", "5. Confusion Matrix — Calibrated Multimodal Fusion Model"),
        ("final_comparison_confusion_matrices.png", "6. Normalized Confusion Matrices — All 3 Models Side-by-Side")
    ]
    
    # Render Primary 6 Plots in a 2-Column Grid
    col_p1, col_p2 = st.columns(2)
    rendered_primary = 0
    
    for idx, (fname, title) in enumerate(primary_plots):
        fpath = os.path.join(plots_dir, fname)
        target_col = col_p1 if (idx % 2 == 0) else col_p2
        with target_col:
            if os.path.exists(fpath):
                st.image(fpath, caption=title, use_container_width=True)
                rendered_primary += 1
            else:
                st.warning(f"⚠️ Primary Chart Missing: `{fname}` in `{plots_dir}`")
                
    st.markdown("---")
    
    # Section D: SECONDARY EXPANDER FOR DETAILED TRAINING DIAGNOSTICS
    with st.expander("🔍 Secondary Visuals: Training Curves & Individual Baseline Confusion Matrices"):
        st.markdown(
            "Detailed training convergence plots, dataset distribution, and individual baseline confusion matrices."
        )
        
        secondary_plots = [
            ("class_distribution.png", "Figure S1: Class & Malignancy Risk Distribution (HAM10000)"),
            ("sample_images_per_class.png", "Figure S2: Sample Lesion Images Per Diagnosis Class"),
            ("cnn_training_curves.png", "Figure S3: EfficientNetB0 Training & Validation Loss/Accuracy"),
            ("fusion_training_curves.png", "Figure S4: Joint Fusion Model Training & Validation Loss/Accuracy"),
            ("confusion_matrix_cnn.png", "Figure S5: Confusion Matrix — Standalone CNN (Images Only)"),
            ("confusion_matrix_random_forest_(metadata_only).png", "Figure S6: Confusion Matrix — Random Forest (Metadata Only)"),
            ("confusion_matrix_xgboost_(metadata_only).png", "Figure S7: Confusion Matrix — XGBoost (Metadata Only)")
        ]
        
        col_s1, col_s2 = st.columns(2)
        for idx, (fname, title) in enumerate(secondary_plots):
            fpath = os.path.join(plots_dir, fname)
            target_col = col_s1 if (idx % 2 == 0) else col_s2
            with target_col:
                if os.path.exists(fpath):
                    st.image(fpath, caption=title, use_container_width=True)
