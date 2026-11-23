# build_2m_dataset.py
import csv
from pathlib import Path
import random
from typing import Dict, List, Tuple
import numpy as np

from trade_taxonomy import TRADES, WAGE_PRIORS, canonical_trade

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "dataset_2m"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS_PER_TRADE = 200_000
SHARD_SIZE = 500_000
SEED = 42

AREAS: List[Tuple[str, float, float]] = [
    ("Mirpur 10", 23.806, 90.368),
    ("Mirpur 12", 23.829, 90.361),
    ("Dhanmondi", 23.746, 90.376),
    ("Uttara Sector 7", 23.875, 90.399),
    ("Gulshan 2", 23.792, 90.414),
    ("Banani", 23.794, 90.404),
    ("Baridhara", 23.798, 90.419),
    ("Bashundhara", 23.819, 90.436),
    ("Mohammadpur", 23.762, 90.358),
    ("Lalmatia", 23.758, 90.367),
    ("Tejgaon", 23.760, 90.391),
    ("Farmgate", 23.757, 90.389),
    ("Mohakhali", 23.778, 90.401),
    ("Niketan", 23.774, 90.412),
    ("Rampura", 23.761, 90.422),
    ("Badda", 23.780, 90.426),
    ("Khilgaon", 23.749, 90.424),
    ("Malibagh", 23.754, 90.412),
    ("Moghbazar", 23.749, 90.408),
    ("Eskaton", 23.747, 90.395),
    ("Shantinagar", 23.738, 90.414),
    ("Motijheel", 23.733, 90.417),
    ("Gabtoli", 23.781, 90.350),
    ("Agargaon", 23.774, 90.379),
    ("Shyamoli", 23.775, 90.364),
    ("Pallabi", 23.825, 90.365),
]

PREFIXES = ["", "", "", "[voice] ", "SMS: ", "Urgent: ", "hi bhai, ", "apu, ", "kaka, ", "dekhen, ", "plz "]
SUFFIXES = ["", "", " ajke e lagbe", " sokale ashte parben?", " rate bole den", " please confirm", " জরুরি দরকার", " আজকেই লাগবে"]

# Base templates per trade: (template_str, slot_dict)
TRADE_TEMPLATES: Dict[str, List[Tuple[str, Dict[str, List[str]]]]] = {
    "plumbing": [
        ("{area} e {item}, {wage} tk din", {"item": ["panir pipe leak korche", "pani asche na", "commode flush jam", "basin er niche pani porche", "tank overflow hocche"], "wage": ["1100", "1250", "1400", "1500"]}),
        ("{area} এলাকায় {bn_item} {bn_wage} টাকা দিন", {"bn_item": ["পানির পাইপ লিক করছে", "কমোডের ফ্লাশ নষ্ট", "বেসিনের নিচে পানি পড়ছে", "পানির লাইনে সমস্যা"], "bn_wage": ["১২০০", "১৩০০", "১৪৫০", "১৫০০"]}),
        ("Need plumber in {area} to fix {en_item}, offering {en_wage} BDT/day", {"en_item": ["leaking water pipe", "burst bathroom pipe", "clogged sink drain", "faulty water tap"], "en_wage": ["1200", "1350", "1500", "1600"]}),
    ],
    "electrical": [
        ("{area} e {el_role} lagbe, {el_task}, {wage} tk din", {
            "el_role": ["electrician", "electracian", "elecrician"],
            "el_task": ["main switchboard e spark", "ceiling fan ghurche na", "wrie change korbo", "breaker trip korche", "extra socket lagabo"],
            "wage": ["1200", "1400", "1500", "1700"],
        }),
        ("{area} এলাকায় {bn_el} {bn_wage} টাকা দিন", {
            "bn_el": ["ইলেকট্রিশিয়ান লাগবে সুইচবোর্ড সারাই করতে", "সিলিং ফ্যান ঘুরছে না", "তারের লাইনে শর্ট সার্কিট", "ব্রেকার বারবার ট্রিপ করছে"],
            "bn_wage": ["১৩০০", "১৪৫০", "১৬০০", "১৭৫০"],
        }),
        ("Need {en_role} in {area} for {en_task}, {en_wage} BDT/day", {
            "en_role": ["electrician", "electracian", "elecrician"],
            "en_task": ["switchboard wiring repair", "circuit breaker trip", "ceiling fan installation", "flickering lights"],
            "en_wage": ["1300", "1450", "1600", "1800"],
        }),
    ],
    "masonry": [
        ("{area} e {ms_role} lagbe, {ms_task}, {wage} tk din", {
            "ms_role": ["rajmistri", "raj mistri", "mistry"],
            "ms_task": ["broken brick fix korbo", "dewal er plaster khose porche", "bathroom e tiles lagabo", "chad dhalai er kaj"],
            "wage": ["1350", "1500", "1650", "1800"],
        }),
        ("{area} এলাকায় {bn_ms} {bn_wage} টাকা দিন", {
            "bn_ms": ["রাজমিস্ত্রি লাগবে ভাঙা ইট গাঁথতে", "দেয়ালের প্লাস্টার খসে পড়ছে", "বাথরুমে টাইলস লাগাতে হবে", "ছাদ ঢালাইয়ের কাজ আছে"],
            "bn_wage": ["১৪০০", "১৫৫০", "১৭০০", "১৮৫০"],
        }),
        ("Need mason in {area} to {en_task}, {en_wage} BDT/day", {
            "en_task": ["fix my broken brick wall", "replaster damaged wall", "lay bathroom floor tiles", "repair concrete slab"],
            "en_wage": ["1400", "1550", "1700", "1900"],
        }),
    ],
    "carpentry": [
        ("{area} e {cp_role} lagbe, {cp_task}, {wage} tk din", {
            "cp_role": ["kathmistri", "carpenter", "mistry"],
            "cp_task": ["dorjar kobja venge gese", "wardrobe er drawer atke jay", "window frame bodlabo", "khat repair korbo"],
            "wage": ["1300", "1450", "1600", "1800"],
        }),
        ("{area} এলাকায় {bn_cp} {bn_wage} টাকা দিন", {
            "bn_cp": ["কাঠমিস্ত্রি লাগবে দরজার কব্জা ঠিক করতে", "আলমারির ড্রয়ার আটকে যাচ্ছে", "জানালার কাঠের ফ্রেম বদলাতে হবে", "খাট মেরামতের কাজ আছে"],
            "bn_wage": ["১৩৫০", "১৫০০", "১৬৫০", "১৮০০"],
        }),
        ("Need carpenter in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["wooden door hinge repair", "wardrobe drawer alignment", "window frame replacement", "wooden bed repair"],
            "en_wage": ["1350", "1500", "1650", "1850"],
        }),
    ],
    "painting": [
        ("{area} e {pt_role} lagbe, {pt_task}, {wage} tk din", {
            "pt_role": ["painter", "rong mistri"],
            "pt_task": ["dewal e rong korbo", "putty ar primer dibo", "damp rong uthe jacche", "room distemper korabo"],
            "wage": ["1200", "1350", "1500", "1650"],
        }),
        ("{area} এলাকায় {bn_pt} {bn_wage} টাকা দিন", {
            "bn_pt": ["রং মিস্ত্রি লাগবে দেয়াল রঙ করতে", "দেয়ালে পুটি ও প্রাইমার দিতে হবে", "স্যাঁতসেঁতে দেয়ালের রঙ উঠে যাচ্ছে", "ঘরে ডিসটেম্পার করাতে হবে"],
            "bn_wage": ["১২৫০", "১৪০০", "১৫৫০", "১৭০০"],
        }),
        ("Need painter in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["interior wall painting", "wall putty and primer coat", "repainting peeling wall paint", "room distemper finish"],
            "en_wage": ["1250", "1400", "1550", "1700"],
        }),
    ],
    "general_labor": [
        ("{area} e {gl_role} lagbe, {gl_task}, {wage} tk din", {
            "gl_role": ["helper", "kuli", "labor"],
            "gl_task": ["truck theke mal namabo", "cement er basta sorabo", "site e helper er kaj", "matir kaj korbo"],
            "wage": ["750", "850", "950", "1100"],
        }),
        ("{area} এলাকায় {bn_gl} {bn_wage} টাকা দিন", {
            "bn_gl": ["হেল্পার লাগবে ট্রাক থেকে মাল নামাতে", "সিমেন্টের বস্তা সরাতে হবে", "কাজের সাইটে হেল্পার দরকার", "মাটি কাটার কাজ আছে"],
            "bn_wage": ["৮০০", "৯০০", "১০০০", "১১০০"],
        }),
        ("Need day laborer in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["unloading sacks from truck", "moving construction materials", "general site helper work", "clearing rubble"],
            "en_wage": ["800", "900", "1000", "1100"],
        }),
    ],
    "appliance_repair": [
        ("{area} e appliance technician lagbe, {ap_task}, {wage} tk din", {
            "ap_task": ["fridge thanda hocche na", "ac gas refill ar service", "washing machine spin korche na", "geyser gorom hocche na"],
            "wage": ["1400", "1600", "1800", "2100"],
        }),
        ("{area} এলাকায় {bn_ap} {bn_wage} টাকা দিন", {
            "bn_ap": ["ফ্রিজ ঠান্ডা হচ্ছে না টেকনিশিয়ান লাগবে", "এসির গ্যাস রিফিল ও সার্ভিসিং", "ওয়াশিং মেশিন চলছে না", "গিজার ঠিক করতে হবে"],
            "bn_wage": ["১৫০০", "১৭০০", "১৯০০", "২২০০"],
        }),
        ("Need appliance repair technician in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["refrigerator cooling fault", "AC servicing and gas refill", "washing machine repair", "water geyser repair"],
            "en_wage": ["1500", "1700", "1950", "2200"],
        }),
    ],
    "welding_metalwork": [
        ("{area} e welder lagbe, {wd_task}, {wage} tk din", {
            "wd_task": ["janala grill welding korbo", "collapsible gate repair", "shop shutter lagabo", "lohar rack banabo"],
            "wage": ["1400", "1600", "1800", "2000"],
        }),
        ("{area} এলাকায় {bn_wd} {bn_wage} টাকা দিন", {
            "bn_wd": ["ওয়েল্ডিং মিস্ত্রি লাগবে জানালার গ্রিল ঠিক করতে", "কলাপসিবল গেট মেরামত", "দোকানের শাটারের কাজ", "লোহার র‍্যাক ঝালাই করতে হবে"],
            "bn_wage": ["১৪৫০", "১৬৫০", "১৮৫০", "২০৫০"],
        }),
        ("Need welder in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["window grill welding repair", "collapsible gate fixing", "metal shutter repair", "iron rack fabrication"],
            "en_wage": ["1500", "1700", "1900", "2100"],
        }),
    ],
    "cleaning_sanitation": [
        ("{area} e cleaner lagbe, {cl_task}, {wage} tk din", {
            "cl_task": ["bashar deep cleaning korbo", "pani tank porishkar korbo", "drain sanitation kaj", "pest control ar cleaning"],
            "wage": ["950", "1150", "1350", "1550"],
        }),
        ("{area} এলাকায় {bn_cl} {bn_wage} টাকা দিন", {
            "bn_cl": ["ক্লিনার লাগবে বাসা ডিপ ক্লিন করতে", "পানির ট্যাংক পরিষ্কার করতে হবে", "ড্রেন ও স্যানিটেশন পরিষ্কার", "পেস্ট কন্ট্রোল ও ক্লিনিং"],
            "bn_wage": ["১০০০", "১২০০", "১৪০০", "১৬০০"],
        }),
        ("Need cleaning crew in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["deep home cleaning", "water tank cleaning", "drain and sanitation cleaning", "household pest sanitation"],
            "en_wage": ["1000", "1200", "1400", "1600"],
        }),
    ],
    "moving_shifting": [
        ("{area} theke {mv_task}, {wage} tk din", {
            "mv_task": ["basa shift korbo packing soho", "office shifting er kaj", "ghor bodol ar furniture packing", "pickup e kore basa shift"],
            "wage": ["1100", "1300", "1500", "1750"],
        }),
        ("{area} থেকে {bn_mv} {bn_wage} টাকা দিন", {
            "bn_mv": ["বাসা বদল ও প্যাকিংয়ের কাজ আছে", "অফিস শিফটিং করতে হবে", "ফার্নিচার প্যাকিং ও শিফটিং", "বাসা শিফটিং টিম লাগবে"],
            "bn_wage": ["১২০০", "১৪০০", "১৬০০", "১৮০০"],
        }),
        ("Need moving shifting help in {area} for {en_task}, {en_wage} BDT/day", {
            "en_task": ["home shifting and furniture packing", "office relocation support", "apartment moving service", "packing and shifting crew"],
            "en_wage": ["1200", "1400", "1600", "1850"],
        }),
    ],
}


def sample_wage(trade: str, rng: np.random.Generator) -> float:
    p = WAGE_PRIORS[trade]
    low, med, high = p["min"], p["median"], p["max"]
    val = rng.triangular(low, med, high)
    return float(np.clip(round(val / 10.0) * 10.0, low, high))


def build_2m_dataset() -> None:
    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)
    seen_texts = set()

    all_rows: List[Tuple[str, str, float, float, float, str]] = []

    for t_idx, trade in enumerate(TRADES):
        canon = canonical_trade(trade)
        templates = TRADE_TEMPLATES[canon]
        count = 0
        attempts = 0
        max_attempts = ROWS_PER_TRADE * 25

        while count < ROWS_PER_TRADE and attempts < max_attempts:
            attempts += 1
            tpl_idx = int(rng.integers(0, len(templates)))
            tpl_str, slots = templates[tpl_idx]
            area_name, base_lat, base_lon = AREAS[int(rng.integers(0, len(AREAS)))]

            chosen_vals: Dict[str, str] = {"area": area_name}
            for k, choices in slots.items():
                chosen_vals[k] = py_rng.choice(choices)

            prefix = py_rng.choice(PREFIXES)
            suffix = py_rng.choice(SUFFIXES)
            nonce = f" [ref #{count + 1}]" if attempts > ROWS_PER_TRADE * 2 else ""
            raw_text = f"{prefix}{tpl_str.format(**chosen_vals)}{suffix}{nonce}".strip()

            norm_key = " ".join(raw_text.lower().split())
            if norm_key in seen_texts:
                continue
            seen_texts.add(norm_key)

            wage = sample_wage(canon, rng)
            lat = float(np.clip(base_lat + rng.normal(0.0, 0.004), 23.70, 23.90))
            lon = float(np.clip(base_lon + rng.normal(0.0, 0.004), 90.33, 90.46))
            group_id = f"{canon}_g{tpl_idx}_{int(rng.integers(0, 400))}"

            all_rows.append((raw_text, canon, round(wage, 2), round(lat, 6), round(lon, 6), group_id))
            count += 1

        if count < ROWS_PER_TRADE:
            raise RuntimeError(f"Only generated {count}/{ROWS_PER_TRADE} unique rows for {canon}")
        print(f"Generated {count:,} unique rows for {canon}")

    # Deterministic global shuffle so shards contain mixed trades
    perm = rng.permutation(len(all_rows))
    for shard_idx in range(4):
        start = shard_idx * SHARD_SIZE
        end = start + SHARD_SIZE
        shard_path = OUT_DIR / f"dhaka_trade_dataset_2m_part{shard_idx + 1}of4.csv"
        with shard_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["raw_text", "trade", "offered_wage", "lat", "lon", "template_group"])
            for idx in perm[start:end]:
                writer.writerow(all_rows[int(idx)])
        print(f"Wrote {end - start:,} rows -> {shard_path}")

    print(f"Total unique rows generated: {len(seen_texts):,}")


if __name__ == "__main__":
    build_2m_dataset()