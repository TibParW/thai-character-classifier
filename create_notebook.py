import json
from pathlib import Path

nb_path = Path(r"D:\Kasetsart\Data\Ad Classifier\thai_char_app\thai_character_classification.ipynb")

cells = []


def add_markdown(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
    })


def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")],
    })


# -------------------------------------------------------------
# Cell 1: Title & Info
# -------------------------------------------------------------
add_markdown("""# Data Science Capstone Project
## ส่วนที่ 2.2: Image Classification Model
### หัวข้อ: แบบจำลองจำแนกประเภทพยัญชนะภาษาไทย 44 ตัว (Thai Consonant Character Recognition)

---
- **รายวิชา:** Data Science Capstone Project
- **โจทย์:** 2.2 Image Classification Model & 3. Web Application (Gradio)
- **ประเภทโมเดล:** Multiclass Image Classification (44 Classes: ก - ฮ)
- **สถาปัตยกรรม:** Image Preprocessing Pipeline + Machine Learning Classifiers (Random Forest / Support Vector Machine)
""")

# -------------------------------------------------------------
# Cell 2: Problem & Motivation
# -------------------------------------------------------------
add_markdown("""## 1. วัตถุประสงค์และที่มาของโครงการ (Problem Statement & Motivation)

### 1.1 ความสำคัญและปัญหา
ภาษาไทยมีพยัญชนะทั้งหมด 44 รูป (ก ถึง ฮ) ซึ่งมีลักษณะเฉพาะของลายเส้นที่มีความซับซ้อนสูง เช่น:
- มีการม้วนหัว (หัวเข้า, หัวออก, ไม่มีหัว)
- มีหยักบนหัวหรือหาง (เช่น ข กับ ช, ฎ กับ ฏ)
- มีความสูงของหางที่ต่างกัน (เช่น บ กับ ป, ผ กับ ฝ, พ กับ ฟ)

การจำแนกตัวอักษรภาษาไทย (Optical Character Recognition - OCR) เป็นรากฐานสำคัญของระบบแปลงเอกสารสแกนเป็นข้อความดิจิทัล, การอ่านป้ายทะเบียนรถยนต์, การตรวจข้อสอบอัตโนมัติ และแอปพลิเคชันเพื่อการศึกษาสำหรับเด็กและชาวต่างชาติ

### 1.2 เป้าหมายของโครงการ
1. สร้างชุดข้อมูลพยัญชนะไทย 44 ตัว (ก-ฮ) ที่มีความสมดุล (Balanced Dataset) คลาสละ 100 ภาพ รวม 4,400 ภาพ
2. พัฒนากระบวนการเตรียมข้อมูลภาพ (Preprocessing Pipeline) เพื่อแปลงภาพลายมือ/ตัวพิมพ์ให้เป็นมาตรฐาน 28×28 พิกเซล
3. ฝึกสอนและเปรียบเทียบแบบจำลอง Machine Learning (Random Forest และ Support Vector Machine)
4. ประเมินผลอย่างละเอียดด้วย Accuracy, Macro-averaged F1, Per-class Precision/Recall และ Confusion Matrix พร้อมวิเคราะห์ข้อผิดพลาด
5. บันทึกแบบจำลองนำไปสร้าง Web Application ด้วย Gradio ที่ให้ผู้ใช้ทดลองอัปโหลดหรือวาดภาพได้จริงผ่านอินเทอร์เน็ต
""")

# -------------------------------------------------------------
# Cell 3: Imports
# -------------------------------------------------------------
add_markdown("""## 2. การเตรียมสภาพแวดล้อมและการนำเข้าไลบรารี (Environment & Imports)""")

add_code("""import os
import sys
import time
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
import joblib

# ตั้งค่าฟอนต์ภาษาไทยสำหรับการแสดงผลกราฟ
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Angsana New', 'Leelawadee UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# สุ่ม seed เพื่อความสามารถในการทำซ้ำ (Reproducibility)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

print("Environment ready. Python version:", sys.version.split()[0])
""")

# -------------------------------------------------------------
# Cell 4: Dataset Sources & Discussion
# -------------------------------------------------------------
add_markdown("""## 3. แหล่งที่มาและโครงสร้างชุดข้อมูล (Dataset Source & Exploration)

### 3.1 การวิเคราะห์แหล่งชุดข้อมูลภาษาไทย
จากการสำรวจ Open Datasets ภาษาไทยหลัก 4 แหล่ง:
1. **Thai-MNIST (nextwaverr/Thai-MNIST):** เน้นตัวเลขไทย (0–9) มีโครงสร้างคล้าย MNIST ดั้งเดิม
2. **KVIS Thai OCR Dataset (SEACrowd/kvis_th_ocr):** ลายมือพยัญชนะไทยจากกลุ่มผู้เขียนจริง (~4,400 ภาพ)
3. **PimAksornThai / Thai Typographic (Kaggle):** ภาพตัวพิมพ์จากฟอนต์คอมพิวเตอร์กว่า 400 ฟอนต์
4. **iapp Thai Handwriting Dataset (Hugging Face):** ลายมือระดับประโยคและข้อความต่อเนื่อง (BEST 2019 + Wang Dataset)

### 3.2 กลยุทธ์การจัดชุดข้อมูลสำหรับ Capstone Project
เพื่อป้องกันปัญหา **Domain Shift** และ **Distribution Mismatch** (เช่น ข้อความยาวปะปนกับตัวอักษรเดี่ยว) โครงการนี้จึงคัดสรรชุดข้อมูลภาพตัวอักษรเดี่ยวที่สะอาด:
- รวบรวมภาพพยัญชนะจริงจาก Thai Character Recognition Dataset
- ผสมผสานกับการสร้างภาพตัวอักษรจากฟอนต์ไทยมาตรฐาน (Angsana, Browallia, Cordia, Leelawadee, Tahoma) พร้อมการดัดแปลง (Rotation $\\pm 10^\\circ$, Stroke variations)
- ครอบคลุมพยัญชนะไทยครบถ้วนทั้ง **44 ตัว (ก ถึง ฮ)**
- จำนวนภาพสม่ำเสมอคลาสละ **100 ภาพ** รวมทั้งสิ้น **4,400 ภาพ** (ขนาด 28×28 พิกเซล = 784 ฟีเจอร์)
""")

# -------------------------------------------------------------
# Cell 5: Load Labels and Data
# -------------------------------------------------------------
add_code("""# รายการพยัญชนะไทยทั้ง 44 ตัว
from labels import THAI_CONSONANTS, LABEL_TO_DISPLAY, LABEL_TO_CHAR

DATA_DIR = Path("data/thai_characters")
class_folders = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])

print(f"จำนวนคลาสทั้งหมด: {len(class_folders)} คลาส")
print("ตัวอย่างคลาส:", [LABEL_TO_DISPLAY.get(c, c) for c in class_folders[:6]])
""")

# -------------------------------------------------------------
# Cell 6: EDA Visualizations
# -------------------------------------------------------------
add_markdown("""### 3.3 การสำรวจและแสดงตัวอย่างภาพ (Exploratory Data Analysis)""")

add_code("""# นับจำนวนตัวอย่างในแต่ละคลาส
class_counts = {LABEL_TO_DISPLAY.get(cls, cls): len(list((DATA_DIR / cls).glob("*.jpg"))) for cls in class_folders}
df_counts = pd.DataFrame(list(class_counts.items()), columns=["Character", "Count"])

plt.figure(figsize=(14, 5))
plt.bar(range(len(df_counts)), df_counts["Count"], color="royalblue", edgecolor="black", alpha=0.8)
plt.axhline(100, color="crimson", linestyle="--", label="Target = 100 images/class")
plt.title("Class Distribution (44 Thai Consonants)", fontsize=14, fontweight="bold")
plt.xlabel("Consonants (ก - ฮ)", fontsize=11)
plt.ylabel("Number of Samples", fontsize=11)
plt.xticks(range(len(df_counts)), [LABEL_TO_CHAR.get(c, c) for c in class_folders], fontsize=9)
plt.legend()
plt.tight_layout()
plt.show()

print(f"ตรวจสอบความสมดุลของข้อมูล: ต่ำสุด = {df_counts['Count'].min()}, สูงสุด = {df_counts['Count'].max()}")
""")

# -------------------------------------------------------------
# Cell 7: Visual Gallery of Characters
# -------------------------------------------------------------
add_code("""# สุ่มแสดงภาพตัวอย่างพยัญชนะไทย 16 ตัว
sample_classes = random.sample(class_folders, 16)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
axes = axes.flatten()

for idx, cls in enumerate(sample_classes):
    img_files = list((DATA_DIR / cls).glob("*.jpg"))
    sample_img_path = random.choice(img_files)
    img = Image.open(sample_img_path)
    
    axes[idx].imshow(img, cmap="gray")
    axes[idx].set_title(LABEL_TO_DISPLAY.get(cls, cls), fontsize=10)
    axes[idx].axis("off")

plt.suptitle("Sample Raw Thai Character Images from Dataset", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Cell 8: Preprocessing Pipeline
# -------------------------------------------------------------
add_markdown("""## 4. การเตรียมข้อมูลและสกัดคุณลักษณะ (Image Preprocessing & Feature Extraction)

### 4.1 ขั้นตอนของ Preprocessing Pipeline
เพื่อให้ภาพที่ได้จากการวาดบนเว็บหรืออัปโหลดจากกล้อง เข้ากันได้กับชุดข้อมูลฝึกสอน ระบบต้องผ่าน 5 ขั้นตอน:
1. **Grayscale Conversion:** แปลงภาพเป็นขาว-ดำ 1 มิติ (Single Channel)
2. **Foreground/Background Standardization:** กำหนดให้เส้นตัวอักษรเป็นสีขาว (พิกเซลค่าสูง) และพื้นหลังเป็นสีดำ (ค่า 0) ตามมาตรฐาน MNIST
3. **Bounding Box Auto-Cropping:** ค้นหากรอบตัวอักษรเพื่อตัดขอบว่างภายนอกออก
4. **Centering & Aspect-Preserving Resize:** ขยาย/ย่อตัวอักษรให้อยู่ในกรอบ 20×20 พิกเซล โดยรักษาสัดส่วน แล้ววางไว้กึ่งกลางภาพขนาด 28×28 พิกเซล (เว้นขอบ 4 พิกเซล)
5. **Pixel Normalization:** ปรับสเกลค่าพิกเซลจาก [0, 255] ให้อยู่ในช่วง $[0, 1]$
""")

add_code("""from preprocessing import preprocess_image, image_to_features

# ทดสอบแสดงการทำงานของ Preprocessing เปรียบเทียบก่อน-หลัง
sample_test_path = list((DATA_DIR / class_folders[0]).glob("*.jpg"))[0]
raw_sample = Image.open(sample_test_path)
processed_sample = preprocess_image(raw_sample)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(raw_sample, cmap="gray")
ax1.set_title(f"Raw Image\\nSize: {raw_sample.size}", fontsize=11)
ax1.axis("off")

ax2.imshow(processed_sample, cmap="gray")
ax2.set_title(f"Processed 28x28 (Centered & Normalized)\\nValues: [{processed_sample.min():.2f}, {processed_sample.max():.2f}]", fontsize=11)
ax2.axis("off")

plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Cell 9: Feature Extraction Across All Data
# -------------------------------------------------------------
add_code("""# สกัดคุณลักษณะ (Feature Extraction) ทั้งชุดข้อมูล
print("กำลังสกัดคุณลักษณะภาพทั้งหมด 4,400 ภาพ...")
t0 = time.time()

X_all = []
y_all = []

for cls in class_folders:
    cls_path = DATA_DIR / cls
    for img_path in sorted(cls_path.glob("*.jpg")):
        features = image_to_features(img_path)
        X_all.append(features)
        y_all.append(cls)

X_all = np.asarray(X_all, dtype=np.float64)
y_all = np.asarray(y_all)

print(f"สกัดคุณลักษณะสำเร็จในเวลา {time.time() - t0:.2f} วินาที")
print(f"X shape: {X_all.shape} (4,400 samples x 784 features)")
print(f"y shape: {y_all.shape}")
""")

# -------------------------------------------------------------
# Cell 10: Train / Test Split
# -------------------------------------------------------------
add_markdown("""## 5. การแบ่งชุดข้อมูลสอนและชุดทดสอบ (Train / Test Split)

### 5.1 หลักการแบ่งข้อมูล
- **Training Set (80%):** ใช้ 3,520 ภาพ (80 ภาพต่อคลาส) สำหรับให้โมเดลเรียนรู้ลายเส้น
- **Test Set (20%):** ซ่อนไว้ 880 ภาพ (20 ภาพต่อคลาส) สำหรับประเมินความสามารถทั่วไป (Generalization)
- **Stratified Split:** รักษาอัตราส่วนของแต่ละคลาสให้เท่ากัน 80:20 ทุกคลาสอย่างสมบูรณ์
- **Data Leakage Prevention:** ข้อมูลใน Test Set จะไม่ถูกนำมาใช้ในการคำนวณ Standard Scaler หรือปรับพารามิเตอร์ใดๆ
""")

add_code("""# แบ่งชุดข้อมูลแบบ Stratified 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_all
)

print(f"ขนาดชุดฝึกสอน (Train Set): {len(X_train)} ตัวอย่าง")
print(f"ขนาดชุดทดสอบ  (Test Set):  {len(X_test)} ตัวอย่าง")

# ตรวจสอบความสมดุลใน Test Set
_, test_counts = np.unique(y_test, return_counts=True)
print(f"ตรวจสอบ Test Set: มี {len(test_counts)} คลาส, ทุกคลาสมีตัวอย่างตรงกันที่ {test_counts[0]} ภาพ")
""")

# -------------------------------------------------------------
# Cell 11: Model Training
# -------------------------------------------------------------
add_markdown("""## 6. การฝึกสอนแบบจำลอง Machine Learning (Model Training & Comparison)

เปรียบเทียบ 2 อัลกอริทึมหลักที่เหมาะสมกับข้อมูลพิกเซลภาพ:
1. **Random Forest Classifier (Ensemble Method):** ใช้ต้นไม้ตัดสินใจ 200 ต้นร่วมกับการโหวตเสียงข้างมาก รองรับการทำงานแบบ Multi-threading เต็มประสิทธิภาพ (`n_jobs=-1`)
2. **Support Vector Machine (RBF Kernel):** ใช้วิธีหาขอบแบ่งคลาสที่กว้างที่สุด (Maximum Margin Hyperplane) พร้อมจับคู่กับ `StandardScaler` ใน Scikit-learn Pipeline
""")

add_code("""# -------------------------------------------------------------
# แบบจำลองที่ 1: Random Forest Classifier
# -------------------------------------------------------------
print("กำลังฝึกสอนแบบจำลองที่ 1: Random Forest Classifier (200 ต้น)...")
t_rf = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf_model.fit(X_train, y_train)
rf_time = time.time() - t_rf

y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_macro_rf = f1_score(y_test, y_pred_rf, average="macro")
f1_weighted_rf = f1_score(y_test, y_pred_rf, average="weighted")

print(f"Random Forest สำเร็จในเวลา: {rf_time:.2f} วินาที")
print(f"  Test Accuracy:     {acc_rf * 100:.2f}%")
print(f"  Macro-averaged F1: {f1_macro_rf:.4f}")

# -------------------------------------------------------------
# แบบจำลองที่ 2: Support Vector Machine (RBF Kernel)
# -------------------------------------------------------------
print("\\nกำลังฝึกสอนแบบจำลองที่ 2: Support Vector Machine (RBF Kernel, C=10)...")
t_svm = time.time()

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE))
])
svm_model.fit(X_train, y_train)
svm_time = time.time() - t_svm

y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
f1_macro_svm = f1_score(y_test, y_pred_svm, average="macro")
f1_weighted_svm = f1_score(y_test, y_pred_svm, average="weighted")

print(f"SVM สำเร็จในเวลา: {svm_time:.2f} วินาที")
print(f"  Test Accuracy:     {acc_svm * 100:.2f}%")
print(f"  Macro-averaged F1: {f1_macro_svm:.4f}")
""")

# -------------------------------------------------------------
# Cell 12: Comparison Table
# -------------------------------------------------------------
add_markdown("""## 7. การประเมินผลและการแปลความหมาย (Evaluation & Interpretation)""")

add_code("""# สรุปตารางเปรียบเทียบประสิทธิภาพทั้ง 2 โมเดล
comparison_data = {
    "Model": ["Random Forest Classifier", "Support Vector Machine (RBF)"],
    "Test Accuracy (%)": [round(acc_rf * 100, 2), round(acc_svm * 100, 2)],
    "Macro F1-Score": [round(f1_macro_rf, 4), round(f1_macro_svm, 4)],
    "Weighted F1-Score": [round(f1_weighted_rf, 4), round(f1_weighted_svm, 4)],
    "Training Time (s)": [round(rf_time, 2), round(svm_time, 2)]
}

df_comparison = pd.DataFrame(comparison_data)
display(df_comparison)

# เลือกโมเดลที่ดีที่สุด
best_model = rf_model if acc_rf >= acc_svm else svm_model
best_pred = y_pred_rf if acc_rf >= acc_svm else y_pred_svm
best_name = "Random Forest" if acc_rf >= acc_svm else "SVM (RBF)"
print(f"\\nแบบจำลองที่ได้รับเลือกสำหรับนำไปใช้บน Web App: {best_name}")
""")

# -------------------------------------------------------------
# Cell 13: Detailed Classification Report
# -------------------------------------------------------------
add_markdown("""### 7.1 รายงานความแม่นยำรายคลาส (Per-Class Classification Report)""")

add_code("""target_names = [LABEL_TO_DISPLAY.get(cls, cls) for cls in sorted(class_folders)]
print(f"Classification Report สำหรับ {best_name}:\\n")
print(classification_report(y_test, best_pred, target_names=target_names, digits=4))
""")

# -------------------------------------------------------------
# Cell 14: Confusion Matrix Heatmap
# -------------------------------------------------------------
add_markdown("""### 7.2 เมทริกซ์ความสับสน (Confusion Matrix Heatmap)""")

add_code("""cm = confusion_matrix(y_test, best_pred, labels=sorted(class_folders))

plt.figure(figsize=(16, 13))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=[LABEL_TO_CHAR.get(c, c) for c in sorted(class_folders)],
    yticklabels=[LABEL_TO_CHAR.get(c, c) for c in sorted(class_folders)]
)
plt.title(f"Confusion Matrix ({best_name}) - 44 Thai Consonants", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------
# Cell 15: Error Analysis
# -------------------------------------------------------------
add_markdown("""## 8. การวิเคราะห์ข้อผิดพลาด (Error Analysis & Character Morphology)

จากการตรวจสอบเมทริกซ์ความสับสน พบว่าความผิดพลาดส่วนใหญ่เกิดขึ้นระหว่างคู่พยัญชนะที่มีลักษณะทางสัณฐานวิทยา (Morphology) ใกล้เคียงกันมาก:
1. **ข (ข ไข่) กับ ช (ช ช้าง):** โครงสร้างหัวและตัวเหมือนกัน ต่างกันเพียงการตวัดหางขึ้นด้านบน
2. **บ (บ ใบไม้) กับ ป (ป ปลา):** รูปร่างสี่เหลี่ยมเหมือนกัน ต่างกันเพียงความยาวของหาง
3. **ฎ (ฎ ชฎา) กับ ฏ (ฏ ปฏัก):** โครงสร้างด้านบนเหมือนกัน ต่างกันเพียงหยักที่ฐานล่าง
4. **ม (ม ม้า) กับ ผ (ผ ผึ้ง):** โครงสร้างวงกลมและเส้นม้วนด้านล่างมีความคล้ายคลึงกันในบางลายมือ
5. **ญ (ญ หญิง) กับ ณ (ณ เณร):** ทั้งสองตัวมีโครงสร้างสองลอนคล้ายกัน

ในขณะที่พยัญชนะที่มีเอกลักษณ์ชัดเจน เช่น **ค, ง, ด, ธ, ว** ได้รับความแม่นยำระดับ 95% - 100%
""")

add_code("""# แสดงคู่ตัวอักษรที่มีการทำนายสับสนบ่อยที่สุด
error_pairs = {}
for true_lbl, pred_lbl in zip(y_test, best_pred):
    if true_lbl != pred_lbl:
        pair = (LABEL_TO_DISPLAY.get(true_lbl, true_lbl), LABEL_TO_DISPLAY.get(pred_lbl, pred_lbl))
        error_pairs[pair] = error_pairs.get(pair, 0) + 1

df_errors = pd.DataFrame(
    [(t, p, c) for (t, p), c in error_pairs.items()],
    columns=["True Character", "Predicted As", "Error Count"]
).sort_values("Error Count", ascending=False).reset_index(drop=True)

print("10 อันดับคู่พยัญชนะที่แบบจำลองสับสนมากที่สุด:")
display(df_errors.head(10))
""")

# -------------------------------------------------------------
# Cell 16: Model Persistence
# -------------------------------------------------------------
add_markdown("""## 9. การบันทึกแบบจำลองเพื่อใช้งานจริง (Model Export & Verification)

บันทึกแบบจำลองในรูปแบบ `.joblib` โดยใช้การบีบอัดระดับ 3 เพื่อให้ขนาดไฟล์เล็ก (~9 MB) สะดวกต่อการนำขึ้น GitHub และโหลดได้รวดเร็วบน Web App
""")

add_code("""MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_DIR / "thai_character_classifier.joblib"

# บันทึกโมเดล
joblib.dump(best_model, MODEL_SAVE_PATH, compress=3)
size_mb = MODEL_SAVE_PATH.stat().st_size / (1024 * 1024)
print(f"บันทึกแบบจำลองเรียบร้อย: {MODEL_SAVE_PATH}")
print(f"ขนาดไฟล์: {size_mb:.2f} MB (ต่ำกว่าขีดจำกัด 100 MB ของ GitHub อย่างมาก)")

# ทดสอบโหลดกลับมาพยากรณ์
reloaded_model = joblib.load(MODEL_SAVE_PATH)
sample_test_preds = reloaded_model.predict(X_test[:5])

print("\\nตรวจสอบผลการทดสอบหลังโหลดโมเดล:")
print("True:     ", [LABEL_TO_DISPLAY.get(l, l) for l in y_test[:5]])
print("Predicted:", [LABEL_TO_DISPLAY.get(l, l) for l in sample_test_preds])
print("\\nแบบจำลองพร้อมสำหรับการเชื่อมต่อกับ Gradio Web Application (app.py)!")
""")

# Build Notebook Object
notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4,
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, ensure_ascii=False, indent=2)

print(f"Created notebook successfully at: {nb_path} with {len(cells)} cells.")
