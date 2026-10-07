"""
generate_dataset.py
-------------------
สคริปต์สำหรับสร้างชุดข้อมูลพยัญชนะไทย 44 ตัว (ก - ฮ) จากฟอนต์คอมพิวเตอร์ (Synthetic Generation)
พร้อมทำ Data Augmentation (สุ่มเอียง สุ่มเลื่อนตำแหน่ง สุ่มขนาด และสุ่มความหนา-บางของเส้น)
เพื่อเติมเต็มชุดข้อมูลให้สมดุลครบ 500 ภาพต่อคลาส (รวม 22,000 ภาพ)
"""

import os
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# นำเข้า Mapping พยัญชนะไทย 44 ตัว
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
from labels import THAI_CONSONANTS

DATASET_DIR = BASE_DIR / "data" / "thai_characters"
FONT_DIR = BASE_DIR / "fonts"
TARGET_PER_CLASS = 500


def get_all_fonts():
    """รวบรวมฟอนต์ทั้งจากโฟลเดอร์ fonts/ ของโปรเจกต์ และฟอนต์ Windows"""
    local_fonts = [str(p) for p in FONT_DIR.glob("*.ttf")]

    windows_fonts = [
        r"C:\Windows\Fonts\tahoma.ttf",
        r"C:\Windows\Fonts\tahomabd.ttf",
        r"C:\Windows\Fonts\leelawad.ttf",
        r"C:\Windows\Fonts\leelawdb.ttf",
        r"C:\Windows\Fonts\LeelaUIb.ttf",
        r"C:\Windows\Fonts\angsana.ttc",
        r"C:\Windows\Fonts\cordia.ttc",
        r"C:\Windows\Fonts\browalia.ttc",
    ]

    all_fonts = [f for f in local_fonts + windows_fonts if os.path.exists(f)]
    print(f"พบไฟล์ฟอนต์ทั้งหมด: {len(all_fonts)} ไฟล์")
    return all_fonts


def generate_synthetic_data():
    """สร้างภาพตัวพิมพ์ ก-ฮ พร้อม Augmentation จนครบ 500 ภาพต่อคลาส"""
    font_pool = get_all_fonts()
    random.seed(42)

    total_created = 0
    print("\nกำลังเริ่มสร้างภาพสังเคราะห์...")

    for folder_name, ch, desc in THAI_CONSONANTS:
        class_folder = DATASET_DIR / folder_name
        class_folder.mkdir(parents=True, exist_ok=True)

        existing_files = [
            f for f in class_folder.iterdir() if f.suffix.lower() in {".jpg", ".png"}
        ]
        needed = TARGET_PER_CLASS - len(existing_files)

        if needed <= 0:
            continue

        for i in range(needed):
            # 1. สุ่มคุณสมบัติตัวอักษร
            font_path = random.choice(font_pool)
            size = random.randint(30, 48)       # สุ่มขนาดฟอนต์ 30-48 pt
            angle = random.uniform(-15, 15)     # สุ่มมุมเอียง -15 ถึง +15 องศา
            dx = random.randint(-5, 5)          # สุ่มเลื่อนตำแหน่งแนวนอน ±5 px
            dy = random.randint(-5, 5)          # สุ่มเลื่อนตำแหน่งแนวตั้ง ±5 px

            # 2. สร้างกระดานดำ 56x56 พิกเซล
            canvas = Image.new("L", (56, 56), color=0)
            draw = ImageDraw.Draw(canvas)

            try:
                font = ImageFont.truetype(font_path, size)
            except Exception:
                font = ImageFont.load_default()

            # คำนวณวางตัวหนังสือตรงกลาง
            bbox = draw.textbbox((0, 0), ch, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (56 - w) // 2 - bbox[0] + dx
            y = (56 - h) // 2 - bbox[1] + dy

            # วาดตัวอักษรสีขาว (fill=255)
            draw.text((x, y), ch, fill=255, font=font)

            # 3. หมุนเอียงภาพ
            if angle != 0:
                canvas = canvas.rotate(angle, resample=Image.Resampling.BILINEAR)

            # 4. สุ่มฟิลเตอร์จำลองชนิดเส้นปากกา (Stroke Variations)
            r = random.random()
            if r < 0.20:
                canvas = canvas.filter(ImageFilter.MaxFilter(3))  # ปากกาหัวหนา
            elif r < 0.35:
                canvas = canvas.filter(ImageFilter.MinFilter(3))  # เส้นบางเฉียบ
            elif r < 0.45:
                canvas = canvas.filter(ImageFilter.GaussianBlur(0.7))  # หัวพู่กันนุ่ม

            # 5. บันทึกไฟล์ภาพลงในโฟลเดอร์ของคลาสนั้นๆ
            fname = f"master_500_{len(existing_files) + i + 1:04d}.jpg"
            canvas.save(class_folder / fname, "JPEG", quality=95)
            total_created += 1

    print(f"\nสร้างภาพสำเร็จทั้งหมด: {total_created} ภาพ")
    total_all = sum(
        len(list(d.glob("*.jpg"))) for d in DATASET_DIR.iterdir() if d.is_dir()
    )
    print(f"จำนวนภาพรวมทุกคลาสในระบบ: {total_all} ภาพ (ครบ {TARGET_PER_CLASS} ภาพ/คลาส)\n")


if __name__ == "__main__":
    generate_synthetic_data()
