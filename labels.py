"""
Thai Consonant Labels and Mappings
44 Thai Consonants (พยัญชนะไทย 44 รูป)
"""

THAI_CONSONANTS = [
    ('01_kor_kai', 'ก', 'ก ไก่'),
    ('02_khor_khai', 'ข', 'ข ไข่'),
    ('03_khor_khuat', 'ฃ', 'ฃ ขวด'),
    ('04_khor_khwai', 'ค', 'ค ควาย'),
    ('05_khor_khon', 'ฅ', 'ฅ คน'),
    ('06_khor_rakhang', 'ฆ', 'ฆ ระฆัง'),
    ('07_ngor_ngu', 'ง', 'ง งู'),
    ('08_jor_jan', 'จ', 'จ จาน'),
    ('09_chor_ching', 'ฉ', 'ฉ ฉิ่ง'),
    ('10_chor_chang', 'ช', 'ช ช้าง'),
    ('11_sor_so', 'ซ', 'ซ โซ่'),
    ('12_chor_choe', 'ฌ', 'ฌ เฌอ'),
    ('13_yor_ying', 'ญ', 'ญ หญิง'),
    ('14_dor_chada', 'ฎ', 'ฎ ชฎา'),
    ('15_tor_patak', 'ฏ', 'ฏ ปฏัก'),
    ('16_thor_than', 'ฐ', 'ฐ ฐาน'),
    ('17_thor_montho', 'ฑ', 'ฑ มณโฑ'),
    ('18_thor_phuthao', 'ฒ', 'ฒ ผู้เฒ่า'),
    ('19_nor_nen', 'ณ', 'ณ เณร'),
    ('20_dor_dek', 'ด', 'ด เด็ก'),
    ('21_tor_tao', 'ต', 'ต เต่า'),
    ('22_thor_thung', 'ถ', 'ถ ถุง'),
    ('23_thor_thahan', 'ท', 'ท ทหาร'),
    ('24_thor_thong', 'ธ', 'ธ ธง'),
    ('25_nor_nu', 'น', 'น หนู'),
    ('26_bor_baimai', 'บ', 'บ ใบไม้'),
    ('27_por_pla', 'ป', 'ป ปลา'),
    ('28_phor_phueng', 'ผ', 'ผ ผึ้ง'),
    ('29_for_fa', 'ฝ', 'ฝ ฝา'),
    ('30_phor_phan', 'พ', 'พ พาน'),
    ('31_for_fan', 'ฟ', 'ฟ ฟัน'),
    ('32_phor_samphao', 'ภ', 'ภ สำเภา'),
    ('33_mor_ma', 'ม', 'ม ม้า'),
    ('34_yor_yak', 'ย', 'ย ยักษ์'),
    ('35_ror_ruea', 'ร', 'ร เรือ'),
    ('36_lor_ling', 'ล', 'ล ลิง'),
    ('37_wor_waen', 'ว', 'ว แหวน'),
    ('38_sor_sala', 'ศ', 'ศ ศาลา'),
    ('39_sor_ruesi', 'ษ', 'ษ ฤๅษี'),
    ('40_sor_suea', 'ส', 'ส เสือ'),
    ('41_hor_hip', 'ห', 'ห หีบ'),
    ('42_lor_chula', 'ฬ', 'ฬ จุฬา'),
    ('43_or_ang', 'อ', 'อ อ่าง'),
    ('44_hor_nokhuk', 'ฮ', 'ฮ นกฮูก'),
]

# Map folder ID to display name e.g. "01_kor_kai" -> "ก (ก ไก่)"
LABEL_TO_DISPLAY = {folder: f"{char} ({desc})" for folder, char, desc in THAI_CONSONANTS}

# Map folder ID to character only e.g. "01_kor_kai" -> "ก"
LABEL_TO_CHAR = {folder: char for folder, char, desc in THAI_CONSONANTS}
