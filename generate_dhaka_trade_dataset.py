#!/usr/bin/env python3
"""
Dhaka informal-sector micro-employment dataset generator (2,000,000 unique rows).

Output (in --out):
  dhaka_trade_dataset_2m_part{1..4}of4.csv        raw_text,trade,offered_wage,lat,lon
  dhaka_trade_dataset_2m_part{1..4}of4_meta.csv   row_id,template_group_id,lang,is_hard_negative  (row-aligned sidecar)
  template_groups.csv                             template_group_id,trade,lang,kind,template_text
  README_splits.md                                StratifiedGroupKFold instructions

Streams to disk in chunks; RAM is dominated by two 64/128-bit hash sets (~150 MB each, not held together).
"""
import argparse, csv, hashlib, os, random, re, resource, time
import numpy as np

TRADES = ["plumbing", "El", "masonry", "carpentry", "painting", "general_labor",
          "appliance_repair", "welding_metalwork", "cleaning_sanitation", "moving_shifting"]
# (min, median, max) BDT/day, same order as TRADES
WAGES = [(700, 1300, 2200), (800, 1450, 2400), (850, 1550, 2500), (800, 1500, 2500),
         (750, 1350, 2200), (500, 900, 1500), (900, 1650, 2800), (850, 1600, 2600),
         (650, 1150, 2000), (750, 1300, 2300)]
LANGS = ["bn", "rom", "en"]
LANG_COUNTS_PER_50K = {"bn": 17500, "rom": 20000, "en": 12500}   # 35% / 40% / 25%
HN_RATE = 0.12   # share of hard-negative / boundary rows
LAT_RNG, LON_RNG = (23.70, 23.90), (90.33, 90.46)
JITTER = 0.008

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
def bn_num(n): return str(n).translate(BN_DIGITS)

# ----------------------------------------------------------------------------- areas
# (en_names, bn_names, lat, lon) -- approximate neighbourhood centroids
AREAS = []
for n, (la, lo) in {1: (23.7956, 90.3537), 2: (23.8070, 90.3640), 10: (23.8069, 90.3687),
                    11: (23.8180, 90.3660), 12: (23.8280, 90.3640)}.items():
    AREAS.append(((f"Mirpur {n}", f"Mirpur-{n}", f"Mirpur {n} no"),
                  (f"মিরপুর {bn_num(n)}", f"মিরপুর-{bn_num(n)}", f"মিরপুর {bn_num(n)} নম্বর"), la, lo))
for s in range(1, 15):   # approximate sector grid inside Uttara
    la = 23.8600 + 0.0075 * ((s - 1) // 4)
    lo = 90.3880 + 0.0070 * ((s - 1) % 4)
    AREAS.append(((f"Uttara Sector {s}", f"Uttara Sec-{s}", f"Uttara sector {s}"),
                  (f"উত্তরা {bn_num(s)} নম্বর সেক্টর", f"উত্তরা সেক্টর {bn_num(s)}"), la, lo))
_OTHER = [
    ("Gulshan 1", "গুলশান ১", 23.7806, 90.4165), ("Gulshan 2", "গুলশান ২", 23.7925, 90.4143),
    ("Banani", "বনানী", 23.7937, 90.4066), ("Baridhara", "বারিধারা", 23.8040, 90.4220),
    ("Bashundhara R/A", "বসুন্ধরা আবাসিক", 23.8190, 90.4310), ("Dhanmondi", "ধানমন্ডি", 23.7461, 90.3742),
    ("Mohammadpur", "মোহাম্মদপুর", 23.7662, 90.3586), ("Lalmatia", "লালমাটিয়া", 23.7550, 90.3680),
    ("Tejgaon", "তেজগাঁও", 23.7590, 90.3920), ("Farmgate", "ফার্মগেট", 23.7561, 90.3872),
    ("Mohakhali", "মহাখালী", 23.7780, 90.4050), ("Niketan", "নিকেতন", 23.7830, 90.4080),
    ("Rampura", "রামপুরা", 23.7610, 90.4250), ("Badda", "বাড্ডা", 23.7805, 90.4260),
    ("Khilgaon", "খিলগাঁও", 23.7490, 90.4310), ("Malibagh", "মালিবাগ", 23.7490, 90.4130),
    ("Moghbazar", "মগবাজার", 23.7490, 90.4040), ("Eskaton", "ইস্কাটন", 23.7450, 90.4030),
    ("Shantinagar", "শান্তিনগর", 23.7370, 90.4130), ("Motijheel", "মতিঝিল", 23.7330, 90.4170),
    ("Jatrabari", "যাত্রাবাড়ী", 23.7100, 90.4350), ("Shyampur", "শ্যামপুর", 23.7080, 90.4470),
    ("Gabtoli", "গাবতলী", 23.7780, 90.3440), ("Agargaon", "আগারগাঁও", 23.7780, 90.3800),
    ("Shyamoli", "শ্যামলী", 23.7740, 90.3650), ("Pallabi", "পল্লবী", 23.8250, 90.3660),
    ("Uttarkhan", "উত্তরখান", 23.8710, 90.4400), ("Dakshinkhan", "দক্ষিণখান", 23.8530, 90.4290),
    ("Keraniganj", "কেরানীগঞ্জ", 23.7050, 90.3500),
    ("Savar", "সাভার", 23.8350, 90.3380),   # real Savar lies west of the lon box; placed at the box's western edge
    ("Kallyanpur", "কল্যাণপুর", 23.7810, 90.3600), ("Kafrul", "কাফরুল", 23.7900, 90.3850),
    ("Wari", "ওয়ারী", 23.7160, 90.4150), ("Lalbagh", "লালবাগ", 23.7190, 90.3890),
    ("Hazaribagh", "হাজারীবাগ", 23.7260, 90.3700), ("Kamrangirchar", "কামরাঙ্গীরচর", 23.7200, 90.3790),
    ("Basabo", "বাসাবো", 23.7360, 90.4310), ("Mugda", "মুগদা", 23.7380, 90.4320),
    ("Cantonment", "ক্যান্টনমেন্ট", 23.8000, 90.4000), ("Aftabnagar", "আফতাবনগর", 23.7700, 90.4500),
    ("Merul Badda", "মেরুল বাড্ডা", 23.7800, 90.4350), ("Elephant Road", "এলিফ্যান্ট রোড", 23.7390, 90.3890),
    ("New Market", "নিউমার্কেট", 23.7340, 90.3850), ("Kawran Bazar", "কারওয়ান বাজার", 23.7510, 90.3930),
    ("Paltan", "পল্টন", 23.7350, 90.4120),
]
for en, bn, la, lo in _OTHER:
    AREAS.append(((en,), (bn,), la, lo))
NA = len(AREAS)
A_LAT = np.array([a[2] for a in AREAS]); A_LON = np.array([a[3] for a in AREAS])

# ----------------------------------------------------------------------------- shared noise pools
PREF = {
 "bn": ["", "", "", "[voice] ", "SMS: ", "জরুরি: ", "ভাই, ", "আপু, ", "কাকা, ", "দেখেন, ", "ভাইয়া, ", "সালাম, ",
        "[voice message] ", "প্লিজ ", "আসসালামু আলাইকুম, ", "Urgent: "],
 "rom": ["", "", "", "[voice] ", "SMS: ", "Urgent: ", "plz ", "hi bhai, ", "apu, ", "kaka, ", "dekhen, ", "bhaiya, ",
         "vai, ", "salam, ", "[voice message] ", "plz bhai, ", "mama, "],
 "en": ["", "", "", "[voice] ", "SMS: ", "Urgent: ", "Plz ", "Hi bhai, ", "Hello apu, ", "Hello, ", "Hi, ",
        "Dear bhaiya, ", "[voice message] ", "Req: ", "Need help: ", "Please "],
}
SUFF = {
 "bn": ["জরুরি", "আজকেই লাগবে", "কাল সকালে আসতে পারবেন?", "রেট বলে দিন", "কথা বলতে চাই", "দ্রুত যোগাযোগ করুন",
        "ধন্যবাদ", "আজ বিকেলে আসুন", "ফোন দিয়েন", "দাম বলুন", "কে আছেন?"],
 "rom": ["urgent lagbe", "rate bole den", "kal sokale ashte parben?", "ajkei lagbe", "call den", "plz jaldi", "thanks",
         "taratari", "ke ache?", "mistri pathan", "dam bolen"],
 "en": ["urgent", "need today", "can you come tomorrow morning?", "tell me your rate", "call me", "asap", "thanks",
        "weekend is ok", "please confirm", "who is available?", "send me your price"],
}
WAGE_FMT = {
 "bn": ["মজুরি {w} টাকা", "দৈনিক {w} টাকা দিব", "{w}৳", "{w}/- দিব", "রেট {w} টাকা", "{w} টাকা দৈনিক",
        "বাজেট {w} টাকা", "দিনে {w} টাকা"],
 "rom": ["rate {w} tk", "{w} taka dibo", "{w}/-", "bdt {w}", "mojuri {w} tk", "daily {w}tk", "{w}৳",
         "dine {w} taka", "budget {w} tk"],
 "en": ["pay {w} BDT/day", "budget {w} tk", "offering {w}/- per day", "BDT {w}", "{w} taka daily",
        "{w}৳ per day", "rate {w} tk", "{w} tk/day"],
}
AREA_FMT = {
 "bn": ["{a}-এ", "{a} এলাকায়", "{a}", "লোকেশন {a}", "{a} থেকে বলছি", "{a} এর কাছে"],
 "rom": ["{a} te", "{a} e", "{a} area", "location {a}", "{a} theke bolchi", "{a} er kache", "{a} e ashte hobe"],
 "en": ["in {a}", "at {a}", "{a} area", "near {a}", "location: {a}", "from {a}", "{a} side"],
}
GEN_D = {
 "bn": ["", "দ্রুত মিস্ত্রি লাগবে", "২ তলার বাসা", "ভাড়া বাসা", "সকালের দিকে আসলে ভালো", "আজকেই করে দিতে হবে",
        "পুরনো বিল্ডিং", "ফ্ল্যাটের ভেতরে কাজ", "অফিসের কাজ", "ছোট কাজ", "একজন লোক হলেই চলবে", "লিফট আছে",
        "গেটে দারোয়ান আছে"],
 "rom": ["", "2 tolar basha", "vara basha", "sokale ashle bhalo", "ajkei kore dite hobe", "purono building",
         "flat er vitore kaj", "office er kaj", "choto kaj", "ekjon lok hole cholbe", "lift ache",
         "gate e darwan ache", "tools shathe anben"],
 "en": ["", "second floor flat", "rented apartment", "morning visit preferred", "must be done today", "old building",
        "work inside the flat", "office job", "small job", "one person is enough", "lift available",
        "guard at the gate", "bring your own tools"],
}
SEPS = {"bn": [", ", "। ", " - ", " | ", " ", "; "], "rom": [", ", ". ", " - ", " | ", " ", "; "],
        "en": [", ", ". ", " - ", " | ", " ", "; "]}
JOIN = [", ", " ", ". ", " - ", " | "]
TYPOS = [("electrician", ["electracian", "elecrician", "electrition", "electritian"]),
         ("wiring", ["wrieing", "wirring", "wireing"]), ("wire", ["wrie", "wier"]),
         ("mistri", ["mistry", "misthri", "mestri"]), ("basay", ["bashay", "bhashay", "basha e"]),
         ("basa", ["basha", "bhasha"]), ("panir", ["paanir", "panir"]), ("pani", ["paani", "pani"]),
         ("plumber", ["plumer", "plomber", "plumbar"]), ("carpenter", ["carpentar", "carpinter"]),
         ("urgent", ["urgnt", "urjent", "urgant"]), ("fridge", ["frdge", "fridg", "frij"]),
         ("painter", ["paintar", "peinter"]), ("technician", ["tecnician", "technishian"]),
         ("helper", ["helpar", "halper"]), ("cleaning", ["cleening", "clining"]), ("leak", ["leek", "lik"]),
         ("problem", ["prblm", "problm"]), ("please", ["plz", "pls"]), ("shifting", ["shifing", "shiftin"]),
         ("welder", ["weldar", "welda"]), ("welding", ["weldin", "wellding"])]

# ----------------------------------------------------------------------------- trade content
# per trade: P = core problem phrases, HN = hard-negative / boundary sentences, TAG = worker-noun phrases
CONTENT = [
# 0 plumbing
{"bn": {"P": ["বাথরুমের পাইপ লিক করছে", "রান্নাঘরের সিংকের নিচে পানি পড়ছে", "ট্যাপ দিয়ে পানি পড়ছে বন্ধ হচ্ছে না",
              "কমোডের ফ্লাশ কাজ করছে না", "ছাদের ট্যাংক থেকে পানি লিক হচ্ছে", "বেসিনের পাইপ ভেঙে গেছে",
              "পানির সমস্যা, লাইনে পানি উঠছে না", "পানির লাইনের পাইপ ফেটে গেছে", "বাথরুমের ড্রেন আটকে গেছে",
              "ফসেট বদলাতে হবে", "বাসায় পানির প্রেশার একদম কম", "পাইপ লিকের কারণে মেঝেতে পানি জমছে"],
        "HN": ["পাইপ ফেটে দেয়ালের রং নষ্ট হচ্ছে, আগে পাইপ মেরামত লাগবে", "ইট ভেঙে নয়, বাথরুমের পানির লাইন লিক করছে",
               "দেয়াল ভিজে যাচ্ছে ভেতরের পাইপ লিক থেকে, প্লাম্বার দরকার", "ছাদের পানির ট্যাংকের পাইপ ফাটা, ঢালাইয়ের কাজ না"],
        "TAG": ["প্লাম্বার লাগবে", "পানির মিস্ত্রি দরকার", "প্লাম্বার ভাই লাগবে"]},
 "rom": {"P": ["amar basay panir somossa", "pipe fete pani porche", "bathroom er pipe leak korche",
               "kitchen sink er niche pani jomche", "tap theke pani porche bondho hocche na",
               "commode er flush kaj korche na", "chader tank theke pani leak hocche", "basin er pipe venge gese",
               "line e pani uthche na", "pani r pressure kom", "bathroom er drain jam hoye gese", "faucet bodlate hobe"],
         "HN": ["pipe fete deyal er rong nosto hocche, age pipe thik korte hobe",
                "deyal bhije jacche pipe leak theke, plumber lagbe",
                "brick na, bathroom er pani'r line leak korche", "plaster khose porche karon pipe leak, pipe ta age dekhen"],
         "TAG": ["plumber lagbe", "plumber bhai lagbe", "pani r mistri lagbe"]},
 "en": {"P": ["water supply problem in my flat", "leaking pipe in the bathroom", "kitchen sink is leaking",
              "tap keeps dripping and will not stop", "commode flush is not working", "overhead water tank is leaking",
              "broken basin pipe needs replacement", "no water coming from the tap line", "burst water pipe in the wall",
              "clogged bathroom drain", "need a new faucet fitted", "very low water pressure in the whole house"],
        "HN": ["pipe burst is damaging the wall paint, fix the pipe first",
               "wall is damp because of a leaking pipe, need a plumber",
               "water leak behind the tiles from the pipe, not a tile job",
               "pipe leak in the ceiling causing plaster to peel, fix the leak"],
        "TAG": ["need a plumber", "plumber required", "plumber needed"]}},
# 1 El
{"bn": {"P": ["সুইচবোর্ডে স্পার্ক হচ্ছে", "বাসার ওয়্যারিং করতে হবে", "শর্ট সার্কিট হয়ে গেছে", "এমসিবি বারবার ট্রিপ করছে",
              "সিলিং ফ্যান ঘুরছে না", "এক্সহস্ট ফ্যান লাগাতে হবে", "কারেন্ট নেই, লাইনে সমস্যা",
              "এলইডি লাইট ও টিউব লাইট লাগাবো", "প্লাগ পয়েন্ট বাড়াতে হবে", "ভোল্টেজ উঠানামা করছে",
              "মিটারের লাইনে সমস্যা", "তার পুড়ে গন্ধ আসছে"],
        "HN": ["ফ্রিজের প্লাগ পয়েন্টে শর্ট সার্কিট, ওয়্যারিং দেখতে হবে", "এসির লাইনে এমসিবি ট্রিপ করছে, তার চেক করতে হবে",
               "সিলিং ফ্যান ঘুরছে না, রেগুলেটর ও সুইচবোর্ডের তার দেখতে হবে",
               "দেয়ালের সকেটে স্পার্ক হচ্ছে, ফ্রিজ লাগানো যাচ্ছে না"],
        "TAG": ["ইলেকট্রিশিয়ান লাগবে", "ইলেকট্রিক মিস্ত্রি দরকার", "কারেন্টের মিস্ত্রি লাগবে"]},
 "rom": {"P": ["switchboard e spark hocche", "basay wiring korte hobe", "short circuit hoye gese", "MCB bar bar trip korche",
               "ceiling fan ghurche na", "exhaust fan lagate hobe", "current nai, line e problem",
               "LED light ar tube light lagabo", "plug point barate hobe", "voltage uthanama korche",
               "meter er line e somossa", "wire pure gondho asche"],
         "HN": ["fridge er plug point e short circuit hoyeche, wiring dekhen", "AC er line e MCB trip korche, wire check korte hobe",
                "fan ghurche na, wiring check korte hobe", "socket e spark hocche, fridge plug lagano jacche na"],
         "TAG": ["electrician lagbe", "electracian pathan", "elecrician bhai lagbe"]},
 "en": {"P": ["sparking switchboard in my room", "house wiring needs to be done", "short circuit in the flat",
              "MCB keeps tripping", "ceiling fan is not rotating", "need an exhaust fan installed", "no current, power line fault",
              "install LED and tube lights", "need extra plug points", "voltage fluctuation all day", "meter line fault",
              "burning smell from the wire"],
        "HN": ["wall socket sparking when I plug in the fridge, need wiring check", "AC line trips the MCB, check the wiring",
               "ceiling fan not spinning, check switch board and wire", "power outlet burnt, fridge cannot be plugged in"],
        "TAG": ["need an electrician", "electracian needed", "electrician required"]}},
# 2 masonry
{"bn": {"P": ["ইট ভেঙে গেছে দেয়ালে", "দেয়ালের প্লাস্টার খসে পড়ছে", "বাথরুমে টাইলস বসাতে হবে", "ফ্লোর টাইলস লাগাব",
              "ছাদ ঢালাই করতে হবে", "রাজমিস্ত্রি লাগবে নতুন দেয়াল তুলতে", "ভাঙা দেয়াল মেরামত", "বাউন্ডারি ওয়ালের ইট গাঁথুনি",
              "দেয়ালে ফাটল ধরেছে", "সিঁড়ির ধাপ ভেঙে গেছে", "রান্নাঘরে ওয়াল টাইলস লাগাব", "ইটের গাঁথুনি করতে হবে"],
        "HN": ["দেয়ালের প্লাস্টার খসে পড়ছে, রং না, আগে প্লাস্টার করতে হবে", "ইট ভেঙে গেছে, রং করার আগে মেরামত দরকার",
               "টাইলস ভেঙেছে পানির সমস্যা না, টাইলস বদলাতে হবে", "পাইপ ঠিক আছে, দেয়ালের ভাঙা অংশ গাঁথুনি করতে হবে"],
        "TAG": ["রাজমিস্ত্রি লাগবে", "মিস্ত্রি আর হেল্পার সহ", "রাজমিস্ত্রি দরকার"]},
 "rom": {"P": ["dewal er ita venge gese", "plaster khose porche", "bathroom e tiles boshate hobe", "floor tiles lagabo",
               "chad dhalai korte hobe", "rajmistri lagbe notun dewal tulte", "vanga dewal marammot",
               "boundary wall er gathuni", "dewale fatol dhorse", "sirir dhap venge gese", "kitchen e wall tiles lagabo",
               "brick work korte hobe"],
         "HN": ["dewal er plaster khose porche, rong na, age plaster korte hobe", "fix my broken brick before painting",
                "pipe thik ache, vanga dewal ta gathte hobe", "tiles venge gese, pani leak na, tiles bodlate hobe"],
         "TAG": ["mistri lagbe", "rajmistri lagbe", "mistry helper shoho lagbe"]},
 "en": {"P": ["broken brick in the wall", "plaster is falling off the wall", "need tiles fitted in the bathroom",
              "floor tile installation", "roof slab casting required", "need a mason to build a new wall",
              "structural wall repair", "boundary wall brickwork", "cracks in the wall",
              "broken staircase steps need repair", "kitchen wall tiles to be fixed", "brickwork needed"],
        "HN": ["plaster crumbling off the wall, repair it before any painting",
               "fix my broken brick near the bathroom, pipes are fine", "cracked tiles need replacing, no plumbing issue",
               "crumbling wall corner needs brick and cement work"],
        "TAG": ["need a mason", "mistri with a helper", "mason required"]}},
# 3 carpentry
{"bn": {"P": ["কাঠের দরজা ঠিক করতে হবে", "জানালার ফ্রেম নষ্ট হয়ে গেছে", "দরজার লক ও কব্জা বদলাব", "আলমারি বানাতে হবে",
              "কাবার্ড মেরামত", "খাটের পায়া ভেঙে গেছে", "টেবিল ও চেয়ার মেরামত", "আসবাবপত্র ঠিক করতে হবে",
              "উইপোকায় কাঠ নষ্ট হয়েছে", "কাঠের শেলফ বানাবো", "দরজা আটকে যাচ্ছে বন্ধ হচ্ছে না", "নতুন কাঠের জানালা লাগাব"],
        "HN": ["লোহার গ্রিল নয়, কাঠের দরজার ফ্রেম বদলাতে হবে", "দরজার রং উঠে গেছে, কিন্তু কাঠ নষ্ট, কাঠমিস্ত্রি লাগবে",
               "আলমারির কাঠে উইপোকা ধরেছে, পোকা মারার না, কাঠ বদল করতে হবে",
               "কাঠের জানালার ফ্রেম পচে গেছে, গ্রিল ঠিক আছে"],
        "TAG": ["কাঠমিস্ত্রি লাগবে", "কাঠের মিস্ত্রি দরকার", "কাঠমিস্ত্রি ভাই লাগবে"]},
 "rom": {"P": ["kather dorja thik korte hobe", "janalar frame noshto hoye gese", "dorjar lock ar hinge bodlabo",
               "almari banate hobe", "cabinet marammot", "khater paya venge gese", "table chair marammot",
               "asbabpotro thik korte hobe", "uipoka kath nosto koreche", "kather shelf banabo",
               "dorja atke jacche bondho hocche na", "notun kather janala lagabo"],
         "HN": ["iron grill na, kather dorjar frame bodlate hobe", "dorjar rong uthe gese kintu kath noshto, kathmistri lagbe",
                "almari te uipoka, kath bodlate hobe", "wooden janalar frame pochche, grill thik ache"],
         "TAG": ["kathmistri lagbe", "carpenter lagbe", "mistri lagbe kather kaj e"]},
 "en": {"P": ["wooden door repair needed", "window frame is damaged", "replace the door lock and hinges", "build a wardrobe",
              "cabinet repair", "bed leg is broken", "table and chair repair", "furniture repair needed",
              "termite damage in wood", "build a wooden shelf", "door is stuck and will not close", "fit a new wooden window"],
        "HN": ["not an iron gate, the wooden door frame needs replacing",
               "door paint is peeling but the wood itself is rotten, need a carpenter",
               "wardrobe wood damaged by termites, needs wood replacement",
               "wooden window frame rotten, the grill is fine"],
        "TAG": ["need a carpenter", "carpenter required", "kathmistri needed"]}},
# 4 painting
{"bn": {"P": ["ঘরের ভেতরের দেয়ালে রং করতে হবে", "বাইরের দেয়ালে রং করাব", "ডিসটেম্পার করতে হবে", "প্লাস্টিক পেইন্ট লাগাব",
              "দেয়ালে পুটি দিতে হবে", "প্রাইমার ও রং লাগবে", "রং উঠে যাচ্ছে নতুন করে রং দরকার", "স্যাঁতসেঁতে দেয়ালে রি-পেইন্ট",
              "পুরো ফ্ল্যাটে রং করাতে হবে", "রঙের কাজ করতে হবে", "ড্রয়িং রুমে নতুন রং", "বেডরুমের দেয়ালে পেইন্ট"],
        "HN": ["দেয়ালের রং উঠে যাচ্ছে, ডিসটেম্পার করতে হবে (পাইপ ঠিক আছে)", "পুটি দিয়ে ফাটা দাগ ঢেকে রং করতে হবে",
               "ভেজা দাগ শুকিয়ে গেছে, এখন শুধু রং লাগবে", "নতুন প্লাস্টার শুকিয়েছে, এখন পুটি ও রং করব"],
        "TAG": ["রং মিস্ত্রি লাগবে", "পেইন্টার দরকার", "রঙের মিস্ত্রি লাগবে"]},
 "rom": {"P": ["ghorer vitor dewale rong korte hobe", "baire dewale rong korabo", "distemper korte hobe",
               "plastic paint lagabo", "dewale putty dite hobe", "primer ar rong lagbe",
               "rong uthe jacche notun kore rong lagbe", "damp dewale repaint", "puro flat e rong korate hobe",
               "rongher kaj korte hobe", "drawing room e notun rong", "bedroom er dewal e paint"],
         "HN": ["rong uthe jacche, distemper korte hobe, pipe thik ache", "putty diye fatol dhekhe rong korte hobe",
                "vejha dag shukiye gese, ekhon shudhu rong lagbe", "notun plaster shukiyeche, ekhon putty ar rong korbo"],
         "TAG": ["rong mistri lagbe", "painter lagbe", "rong er mistri lagbe"]},
 "en": {"P": ["interior wall painting needed", "exterior wall painting", "need distemper done", "apply plastic paint",
              "wall putty work required", "primer and paint needed", "paint is peeling and needs repainting",
              "repaint damp-stained walls", "paint the whole flat", "painting job for my apartment",
              "fresh paint for the drawing room", "bedroom wall paint"],
        "HN": ["paint peeling off, only distemper needed, pipes are fine", "cover hairline cracks with putty and repaint",
               "damp stain has dried, just needs repainting", "fresh plaster is dry, now need putty and paint"],
        "TAG": ["need a painter", "painter required", "paint mistri needed"]}},
# 5 general_labor
{"bn": {"P": ["হেল্পার লাগবে সারাদিনের জন্য", "কুলি লাগবে মাল নামাতে", "ট্রাক থেকে সিমেন্টের বস্তা নামাতে হবে",
              "সাইটে কাজের হেল্পার দরকার", "ভাঙা ইট ও ময়লা সরাতে হবে", "বালু ও ইট টানার লোক লাগবে", "দিনমজুর লাগবে",
              "নির্মাণ সাইটে মাল তোলার কাজ", "ট্রাক থেকে মাল নামানো", "বস্তা তোলা নামানোর লোক", "মিস্ত্রির সাথে হেল্পার"],
        "HN": ["বাসা বদল না, শুধু ট্রাক থেকে সিমেন্ট নামানোর লোক লাগবে", "আসবাব প্যাকিং না, সাইটে মাল টানার হেল্পার লাগবে",
               "রাজমিস্ত্রি আছে, শুধু হেল্পার লাগবে", "ইট ভাঙা নয়, ভাঙা ইটের স্তূপ সরানোর লোক লাগবে"],
        "TAG": ["হেল্পার লাগবে", "২ জন লোক লাগবে", "কুলি লাগবে"]},
 "rom": {"P": ["helper lagbe sharadin er jonno", "coolie lagbe mal namate", "truck theke cement er bosta namate hobe",
               "site e kajer helper dorkar", "vanga ita ar debris soranor lok lagbe", "bali ar ita tanar lok lagbe",
               "dinmojur lagbe", "construction site e mal tolar kaj", "truck theke mal namano", "bosta tola namanor lok",
               "mistri r sathe helper"],
         "HN": ["basa bodol na, shudhu truck theke cement namanor lok lagbe",
                "furniture packing na, site e mal tanar helper lagbe", "mistri ache, shudhu helper lagbe",
                "ita vanga na, vanga itar stup soranor lok lagbe"],
         "TAG": ["helper lagbe", "2 jon lok lagbe", "coolie lagbe"]},
 "en": {"P": ["need a day helper", "labourer needed to unload goods", "unload cement sacks from a truck",
              "construction site helper required", "clear broken brick debris", "need men to carry sand and bricks",
              "daily labour required", "lifting work at a building site", "truck unloading crew",
              "loading and unloading of sacks", "helper to assist the mason"],
        "HN": ["not a house shift, just men to unload cement from a truck",
               "no furniture packing, site helper to carry goods needed", "mason is already there, only a helper needed",
               "not brickwork, just someone to haul away broken brick pile"],
        "TAG": ["need a helper", "2 men required", "labour needed"]}},
# 6 appliance_repair
{"bn": {"P": ["ফ্রিজ ঠান্ডা হচ্ছে না", "ফ্রিজের কম্প্রেসর নষ্ট", "এসি ঠান্ডা করছে না", "এসি সার্ভিসিং করতে হবে",
              "এসির গ্যাস রিফিল লাগবে", "ওয়াশিং মেশিন ঘুরছে না", "ওয়াশিং মেশিনের মোটর সমস্যা", "মাইক্রোওয়েভ গরম করছে না",
              "গিজার কাজ করছে না", "গিজারে পানি গরম হচ্ছে না", "ফ্রিজ থেকে পানি পড়ছে", "এসি থেকে পানি পড়ছে আর আওয়াজ করছে"],
        "HN": ["ফ্রিজ চলছে কিন্তু ঠান্ডা হচ্ছে না, ওয়্যারিং ঠিক আছে", "এসির গ্যাস শেষ, তার বা সুইচ ঠিক আছে",
               "ওয়াশিং মেশিনের মোটর পুড়েছে, বাসার লাইন ঠিক আছে", "গিজার চালু হয় কিন্তু পানি গরম হয় না, পাইপ ঠিক আছে"],
        "TAG": ["ফ্রিজ মেকানিক লাগবে", "এসি টেকনিশিয়ান লাগবে", "মেকানিক দরকার"]},
 "rom": {"P": ["fridge thanda hocche na", "fridge er compressor noshto", "ac thanda korche na", "ac servicing korte hobe",
               "ac er gas refill lagbe", "washing machine ghurche na", "washing machine er motor somossa",
               "microwave gorom korche na", "geyser kaj korche na", "geyser e pani gorom hocche na",
               "fridge theke pani porche", "ac theke pani porche ar awaj korche"],
         "HN": ["fridge cholche kintu thanda hocche na, wiring thik ache", "ac er gas shesh, wire ar switch thik ache",
                "washing machine er motor pure gese, basar line thik ache",
                "geyser on hoy kintu pani gorom hoy na, pipe thik ache"],
         "TAG": ["ac mechanic lagbe", "fridge mistri lagbe", "technician lagbe"]},
 "en": {"P": ["fridge is not cooling", "fridge compressor problem", "AC is not cooling", "AC servicing required",
              "AC gas refill needed", "washing machine not spinning", "washing machine motor issue",
              "microwave not heating", "geyser not working", "water heater is not heating", "fridge leaking water",
              "AC dripping water and making noise"],
        "HN": ["fridge runs but does not cool, house wiring is fine", "AC gas is empty, wiring and switch are fine",
               "washing machine motor burnt, home power line is fine", "geyser turns on but water stays cold, plumbing is fine"],
        "TAG": ["need a technician", "AC mechanic required", "appliance technician needed"]}},
# 7 welding_metalwork
{"bn": {"P": ["লোহার গ্রিল বানাতে হবে", "জানালার গ্রিল লাগাব", "কলাপসিবল গেট বানাতে হবে", "দোকানের শাটার মেরামত",
              "ওয়েল্ডিং এর কাজ আছে", "স্টিলের সিঁড়ির রেলিং বানাব", "লোহার র‍্যাক বানাতে হবে", "গেটের তালা ও রড ওয়েল্ডিং",
              "স্টিল ফ্যাব্রিকেশনের কাজ", "ভাঙা গ্রিল ঝালাই করতে হবে", "বারান্দায় সেফটি গ্রিল", "মেইন গেট রিপেয়ার"],
        "HN": ["কাঠের নয়, লোহার জানালার গ্রিল বানাতে হবে", "কাঠমিস্ত্রি না, দোকানের স্টিল শাটার ওয়েল্ডিং লাগবে",
               "দরজার কাঠ ঠিক আছে, লোহার কলাপসিবল গেট লাগাব", "রং না, গ্রিল ভেঙেছে ঝালাই লাগবে"],
        "TAG": ["ওয়েল্ডিং মিস্ত্রি লাগবে", "লোহার মিস্ত্রি দরকার", "ওয়েল্ডার লাগবে"]},
 "rom": {"P": ["lohar grill banate hobe", "janalar grill lagabo", "collapsible gate banate hobe", "dokaner shutter marammot",
               "welding er kaj ache", "steel er sirir railing banabo", "lohar rack banate hobe", "gate er rod welding",
               "steel fabrication er kaj", "vanga grill jhalai korte hobe", "barandaya safety grill", "main gate repair"],
         "HN": ["kather na, lohar janalar grill banate hobe", "kathmistri na, dokaner steel shutter welding lagbe",
                "dorjar kath thik ache, lohar collapsible gate lagabo", "rong na, grill venge gese jhalai lagbe"],
         "TAG": ["welding mistri lagbe", "welder lagbe", "lohar mistri lagbe"]},
 "en": {"P": ["need an iron window grill", "fit window grills", "build a collapsible gate", "shop shutter repair",
              "welding work available", "make a steel stair railing", "iron rack fabrication", "gate rod welding",
              "steel fabrication job", "broken grill needs welding", "safety grill for the balcony", "main gate repair"],
        "HN": ["not wooden, need an iron window grill made", "not a carpenter job, steel shop shutter needs welding",
               "door wood is fine, need an iron collapsible gate", "not painting, the grill is broken and needs welding"],
        "TAG": ["need a welder", "welding mistri required", "welder needed"]}},
# 8 cleaning_sanitation
{"bn": {"P": ["পুরো বাসা ডিপ ক্লিন করাতে হবে", "অফিস ডিপ ক্লিনিং", "ছাদের ট্যাংক পরিষ্কার করতে হবে",
              "আন্ডারগ্রাউন্ড ট্যাংক পরিষ্কার", "সেপটিক ট্যাংক পরিষ্কার করাব", "স্যুয়ারেজ লাইন পরিষ্কার",
              "তেলাপোকা মারার স্প্রে লাগবে", "উইপোকা ও পোকামাকড় দমন", "পেস্ট কন্ট্রোল করতে হবে",
              "রান্নাঘর ও বাথরুম ডিপ ক্লিন", "ফ্ল্যাট হ্যান্ডওভারের আগে ক্লিনিং", "ড্রেনের নর্দমা পরিষ্কার"],
        "HN": ["ট্যাংকের পাইপ লিক নয়, ট্যাংকের ভেতর ময়লা পরিষ্কার করতে হবে",
               "উইপোকা কাঠের ক্ষতি করেনি, পোকা মারার ট্রিটমেন্ট লাগবে", "নতুন বাথরুম ফিটিং না, বাথরুম ডিপ ক্লিন লাগবে",
               "বাসা বদল হয়ে গেছে, এখন শুধু ডিপ ক্লিন লাগবে"],
        "TAG": ["ক্লিনার লাগবে", "ক্লিনিং টিম লাগবে", "পরিষ্কারের লোক দরকার"]},
 "rom": {"P": ["puro basa deep clean korate hobe", "office deep cleaning", "chader tank porishkar korte hobe",
               "underground tank porishkar", "septic tank porishkar korabo", "sewer line porishkar",
               "telapoka marar spray lagbe", "uipoka ar pokamakor domon", "pest control korte hobe",
               "kitchen ar bathroom deep clean", "flat handover er age cleaning", "drain er nordoma porishkar"],
         "HN": ["tank er pipe leak na, tank er vitor moila porishkar korte hobe",
                "uipoka kath nosto kore ni, pest treatment lagbe", "notun bathroom fitting na, bathroom deep clean lagbe",
                "basa bodol hoye gese, ekhon shudhu deep clean lagbe"],
         "TAG": ["cleaner lagbe", "cleaning team lagbe", "safai kormi lagbe"]},
 "en": {"P": ["full home deep cleaning", "office deep cleaning", "overhead water tank cleaning",
              "underground water tank cleaning", "septic tank cleaning", "sewer line cleaning",
              "cockroach control spray needed", "termite and pest control", "pest control service",
              "deep clean kitchen and bathroom", "cleaning before flat handover", "clean the drain and sewage channel"],
        "HN": ["tank pipe is not leaking, the inside of the tank needs cleaning",
               "termites have not damaged wood, only pest treatment needed", "no new fittings, just a deep clean of the bathroom",
               "shifting is done, now only a deep clean is needed"],
        "TAG": ["need a cleaner", "cleaning crew required", "cleaners needed"]}},
# 9 moving_shifting
{"bn": {"P": ["বাসা বদল করব", "পুরো বাসা শিফটিং করতে হবে", "অফিস শিফটিং এর কাজ", "আসবাবপত্র প্যাকিং করতে হবে",
              "ট্রাকে মালামাল তুলে নতুন বাসায় নিতে হবে", "ফার্নিচার র‍্যাপিং ও প্যাকিং", "পিকআপ ও শিফটিং টিম লাগবে",
              "ঘর শিফট করব পুরনো বাসা থেকে", "ফ্ল্যাট ছেড়ে নতুন ফ্ল্যাটে মাল নেওয়া", "ফ্রিজ আলমারি সহ বাসা শিফট",
              "প্যাকিং ও আনপ্যাকিং সহ শিফটিং"],
        "HN": ["সিমেন্টের বস্তা নামানো নয়, বাসার আসবাব প্যাকিং করে শিফট করতে হবে", "সাইটের হেল্পার না, বাসা বদলের টিম লাগবে",
               "ফ্রিজ মেরামত নয়, ফ্রিজ নতুন বাসায় শিফট করতে হবে", "কাঠের আলমারি ঠিক আছে, শুধু প্যাকিং করে সরাতে হবে"],
        "TAG": ["শিফটিং টিম লাগবে", "প্যাকার লাগবে", "শিফটিং এর লোক লাগবে"]},
 "rom": {"P": ["basa bodol korbo", "puro basa shifting korte hobe", "office shifting er kaj", "asbabpotro packing korte hobe",
               "truck e mal tule notun basay nite hobe", "furniture wrapping ar packing", "pickup ar shifting team lagbe",
               "ghor shift korbo purono basa theke", "flat chere notun flat e mal neya", "fridge almari shoho basa shift",
               "packing unpacking shoho shifting"],
         "HN": ["cement bosta namano na, basar furniture packing kore shift korte hobe",
                "site helper na, basa bodoler team lagbe", "fridge marammot na, fridge notun basay shift korte hobe",
                "kather almari thik ache, shudhu packing kore soranor kaj"],
         "TAG": ["shifting er lok lagbe", "packers lagbe", "shifting team lagbe"]},
 "en": {"P": ["house shifting required", "full home relocation", "office relocation job", "pack household furniture",
              "load goods onto a truck and move to new flat", "furniture wrapping and packing",
              "need a pickup truck and shifting crew", "moving out of my old flat", "shift belongings to a new apartment",
              "move the whole house including fridge and wardrobe", "shifting with packing and unpacking"],
        "HN": ["not unloading cement, need to pack home furniture and shift", "not a site helper, need a house-shifting team",
               "not a fridge repair, just move the fridge to the new flat", "wardrobe is fine, only needs packing and moving"],
        "TAG": ["need movers", "shifting crew required", "packers needed"]}},
]

DLIST = [[GEN_D[L] + CONTENT[t][L]["TAG"] for L in LANGS] for t in range(10)]


# ----------------------------------------------------------------------------- helpers
def norm(s): return " ".join(s.lower().split())
def h64(s): return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")
def h128(s): return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=16).digest(), "little")

def char_typo(s, rnd):
    ws = s.split(" ")
    idx = [i for i, w in enumerate(ws) if len(w) > 4 and w.isascii() and w.isalpha()]
    if not idx: return s
    i = rnd.choice(idx); w = ws[i]; j = rnd.randrange(1, len(w) - 2)
    ws[i] = (w[:j] + w[j + 1] + w[j] + w[j + 2:]) if rnd.random() < 0.6 else (w[:j] + w[j + 1:])
    return " ".join(ws)

def gid_of(ti, li, kind_hn, k): return ti * 1000 + li * 100 + (50 + k if kind_hn else k)

def make_text(ti, li, area_i, w, rnd, trade_label_unused=None):
    L = LANGS[li]; C = CONTENT[ti][L]; D = DLIST[ti][li]
    hn = rnd.random() < HN_RATE
    if hn:
        k = rnd.randrange(len(C["HN"])); core = C["HN"][k]; p_d = 0.5
    else:
        k = rnd.randrange(len(C["P"])); core = C["P"][k]; p_d = 0.85
    if rnd.random() < p_d:
        d = rnd.choice(D)
        if d: core += rnd.choice(SEPS[L]) + d
    if L != "bn":
        for key, vs in TYPOS:
            if key in core and rnd.random() < 0.35:
                core = core.replace(key, rnd.choice(vs), 1)
        if rnd.random() < 0.07: core = char_typo(core, rnd)
        if L == "en" and rnd.random() < 0.5: core = core[0].upper() + core[1:]
    text = rnd.choice(PREF[L]) + core
    ex = []
    if rnd.random() < 0.55:
        en_n, bn_n = AREAS[area_i][0], AREAS[area_i][1]
        a = rnd.choice(bn_n) if L == "bn" else rnd.choice(en_n)
        if L == "rom": a = a.lower()
        ex.append(rnd.choice(AREA_FMT[L]).format(a=a))
    if rnd.random() < 0.60:
        ws = str(w)
        if L == "bn" and rnd.random() < 0.5: ws = ws.translate(BN_DIGITS)
        ex.append(rnd.choice(WAGE_FMT[L]).format(w=ws))
    if rnd.random() < 0.65: ex.append(rnd.choice(SUFF[L]))
    rnd.shuffle(ex)
    for e in ex: text += rnd.choice(JOIN) + e
    return text, gid_of(ti, li, hn, k), hn


def write_group_table(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["template_group_id", "trade", "lang", "kind", "template_text"])
        for ti in range(10):
            for li, L in enumerate(LANGS):
                for k, t in enumerate(CONTENT[ti][L]["P"]): wr.writerow([gid_of(ti, li, False, k), TRADES[ti], L, "core", t])
                for k, t in enumerate(CONTENT[ti][L]["HN"]): wr.writerow([gid_of(ti, li, True, k), TRADES[ti], L, "hard_negative", t])


README = """# Splitting without template leakage

Each row has a `template_group_id` (in the `*_meta.csv` sidecar, row-aligned with the main CSV).
Rows sharing a group id were built from the same base problem sentence (trade x language x template),
differing only in prefix / area / wage / typos / filler. Keep every group on ONE side of a split.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

parts = [pd.read_csv(f"dhaka_trade_dataset_2m_part{i}of4.csv") for i in range(1, 5)]
metas = [pd.read_csv(f"dhaka_trade_dataset_2m_part{i}of4_meta.csv") for i in range(1, 5)]
df = pd.concat([pd.concat(parts, ignore_index=True), pd.concat(metas, ignore_index=True)], axis=1)

# 10 folds -> fold 0 = test (10%), fold 1 = val (10%), rest = train (80%)
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(sgkf.split(df, df["trade"], groups=df["template_group_id"]))
test_idx, val_idx = folds[0][1], folds[1][1]
train_idx = sorted(set(range(len(df))) - set(test_idx) - set(val_idx))
```
Notes: group id = trade*1000 + lang_index*100 + template index (hard negatives offset by +50);
`template_groups.csv` maps ids to the template text. `lang` and `is_hard_negative` columns let you report
per-language / boundary-case metrics. Do not use TF-IDF vocab fit on the full data before splitting.
"""


# ----------------------------------------------------------------------------- generation
def generate(out, seed, rows_per_shard=500_000, shards=4):
    os.makedirs(out, exist_ok=True)
    rnd = random.Random(seed); nprng = np.random.default_rng(seed)
    lo = np.array([w[0] for w in WAGES]); md = np.array([w[1] for w in WAGES]); hi = np.array([w[2] for w in WAGES])
    per_trade = rows_per_shard // 10
    seen = set(); t0 = time.time()
    for sh in range(1, shards + 1):
        seq = np.repeat(np.arange(10), per_trade); nprng.shuffle(seq)
        lang_arr = []
        for _ in range(10):
            a = np.concatenate([np.full(LANG_COUNTS_PER_50K[L] * per_trade // 50_000, i) for i, L in enumerate(LANGS)])
            nprng.shuffle(a); lang_arr.append(a.tolist())
        ptr = [0] * 10
        base = f"dhaka_trade_dataset_2m_part{sh}of{shards}"
        with open(os.path.join(out, base + ".csv"), "w", newline="", encoding="utf-8") as f, \
             open(os.path.join(out, base + "_meta.csv"), "w", newline="", encoding="utf-8") as fm:
            wr = csv.writer(f); wm = csv.writer(fm)
            wr.writerow(["raw_text", "trade", "offered_wage", "lat", "lon"])
            wm.writerow(["row_id", "template_group_id", "lang", "is_hard_negative"])
            CH = 50_000
            for s in range(0, rows_per_shard, CH):
                tr = seq[s:s + CH]; n = len(tr)
                u = nprng.beta(2, 2, n); t = 2 * u - 1
                w = np.where(t > 0, md[tr] + t * (hi[tr] - md[tr]), md[tr] + t * (md[tr] - lo[tr]))
                r50 = nprng.random(n) < 0.6
                w = np.where(r50, np.round(w / 50) * 50, np.round(w / 10) * 10)
                w = np.clip(w, lo[tr], hi[tr]).astype(int)
                ar = nprng.integers(0, NA, n)
                lat = np.clip(A_LAT[ar] + nprng.uniform(-JITTER, JITTER, n), *LAT_RNG)
                lon = np.clip(A_LON[ar] + nprng.uniform(-JITTER, JITTER, n), *LON_RNG)
                trl, wl, arl, latl, lonl = tr.tolist(), w.tolist(), ar.tolist(), lat.tolist(), lon.tolist()
                rows, metas = [], []
                for k in range(n):
                    ti = trl[k]; li = lang_arr[ti][ptr[ti]]; ptr[ti] += 1
                    for _ in range(1000):
                        text, gid, hn = make_text(ti, li, arl[k], wl[k], rnd)
                        key = h64(norm(text))
                        if key not in seen: seen.add(key); break
                    else:
                        raise RuntimeError("could not find unique text; template space exhausted")
                    rows.append([text, TRADES[ti], wl[k], f"{latl[k]:.6f}", f"{lonl[k]:.6f}"])
                    metas.append([(sh - 1) * rows_per_shard + s + k, gid, LANGS[li], int(hn)])
                wr.writerows(rows); wm.writerows(metas)
        print(f"shard {sh}/{shards} done  total_unique={len(seen):,}  {time.time() - t0:.0f}s", flush=True)
    del seen
    write_group_table(os.path.join(out, "template_groups.csv"))
    with open(os.path.join(out, "README_splits.md"), "w", encoding="utf-8") as f: f.write(README)


# ----------------------------------------------------------------------------- verification
WNUM = re.compile(r"(?<![0-9০-৯])([0-9০-৯]{3,4})(?![0-9০-৯])")

def verify(out, shards=4, rows_per_shard=500_000):
    seen = set(); total = 0; nulls = 0; dup = 0; mism = 0; with_w = 0
    cnt = {t: 0 for t in TRADES}; wages = {t: [] for t in TRADES}
    lang_cnt = {L: 0 for L in LANGS}; hn_cnt = 0; shard_trade = []
    lat_min = lon_min = 1e9; lat_max = lon_max = -1e9
    for sh in range(1, shards + 1):
        base = os.path.join(out, f"dhaka_trade_dataset_2m_part{sh}of{shards}")
        sc = {t: 0 for t in TRADES}; n_sh = 0
        with open(base + ".csv", encoding="utf-8", newline="") as f, open(base + "_meta.csv", encoding="utf-8", newline="") as fm:
            rd = csv.reader(f); rm = csv.reader(fm)
            assert next(rd) == ["raw_text", "trade", "offered_wage", "lat", "lon"]; next(rm)
            for row, m in zip(rd, rm):
                n_sh += 1; total += 1
                if len(row) != 5 or any(x == "" for x in row): nulls += 1; continue
                text, trade, wg, la, lo_ = row; wg = int(wg)
                k = h128(norm(text))
                if k in seen: dup += 1
                seen.add(k)
                cnt[trade] += 1; sc[trade] += 1; wages[trade].append(wg)
                lang_cnt[m[2]] += 1; hn_cnt += int(m[3])
                la, lo_ = float(la), float(lo_)
                lat_min = min(lat_min, la); lat_max = max(lat_max, la); lon_min = min(lon_min, lo_); lon_max = max(lon_max, lo_)
                nums = WNUM.findall(text)
                if nums:
                    with_w += 1
                    if any(int(x.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))) != wg for x in nums): mism += 1
        shard_trade.append((n_sh, sc))
    print("\n================ VERIFICATION REPORT ================")
    print(f"total rows                : {total:,}  (expected 2,000,000) -> {'OK' if total == 2_000_000 else 'FAIL'}")
    print(f"nunique(raw_text) (norm.) : {len(seen):,} -> {'OK' if len(seen) == total == 2_000_000 and dup == 0 else 'FAIL'}   duplicates={dup}")
    print(f"null / empty fields       : {nulls} -> {'OK' if nulls == 0 else 'FAIL'}")
    print(f"rows per trade            : " + ", ".join(f"{t}={cnt[t]:,}" for t in TRADES))
    print(f"  exact 200,000 per trade -> {'OK' if all(c == 200_000 for c in cnt.values()) else 'FAIL'}")
    print("per-shard rows / per-trade  : " + "; ".join(f"part{i+1}={n:,}/{set(sc.values())}" for i, (n, sc) in enumerate(shard_trade)))
    print(f"language mix              : " + ", ".join(f"{L}={lang_cnt[L] / total:.1%}" for L in LANGS))
    print(f"hard-negative rows        : {hn_cnt:,} ({hn_cnt / total:.1%})")
    print(f"rows quoting a wage in text: {with_w:,}; wage mismatches vs offered_wage: {mism} -> {'OK' if mism == 0 else 'FAIL'}")
    print(f"lat range {lat_min:.4f}..{lat_max:.4f}  lon range {lon_min:.4f}..{lon_max:.4f}")
    print(f"\n{'trade':22s}{'min':>7s}{'median':>8s}{'max':>7s}   target(min/med/max)")
    for i, t in enumerate(TRADES):
        a = np.array(wages[t]); print(f"{t:22s}{a.min():7d}{int(np.median(a)):8d}{a.max():7d}   {WAGES[i]}")
    print(f"\npeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--electrical-label", default="El", help="label for the electrical trade (spec text says 'El' in the taxonomy, 'electrical' elsewhere)")
    ap.add_argument("--verify-only", action="store_true"); ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    TRADES[1] = a.electrical_label
    if not a.verify_only: generate(a.out, a.seed)
    if not a.no_verify: verify(a.out)
