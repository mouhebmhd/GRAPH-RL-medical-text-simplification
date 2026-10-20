"""Patient-facing glossary: find medical terms in text and attach a short definition + extra info.

Three sources, in order of trust:
  1. BUILT-IN glossary (this file): short, general, hand-written. Body parts, medicines, conditions, drug
     classes, abbreviations. No dosing advice - only what it is, what it is for, and one general tip.
  2. KNOWN NAME with no entry: any name in CRITICAL_TERMS (the clinician drug list, extendable with
     SIPG_CRITICAL_TERMS) is flagged as a medicine even when this file has no text for it.
  3. WORD-SHAPE guess: words that *look* medical (-itis, -emia, -pril, -statin ...). Flagged "unverified";
     the dashboard can ask the local LLM to explain them on demand (see llm.OpenAICompatLLM.explain).

This is general education, not medical advice. Extend GLOSSARY_FILE (JSON list of entries) without code changes:
    SIPG_GLOSSARY=/path/extra.json   ->  [{"keys": ["tachypnea"], "type": "condition",
                                           "definition": "...", "extra": "..."}]
"""
from __future__ import annotations
import json, os, re
from dataclasses import dataclass, asdict
from .lexicon import CRITICAL_TERMS

TYPE_LABELS = {"body": "Body part", "medicine": "Medicine", "class": "Type of medicine",
               "condition": "Condition or symptom", "abbr": "Abbreviation", "test": "Test or procedure",
               "other": "Medical term"}
DRUG_TIP = "Take it exactly as your clinician or pharmacist told you, and ask them before changing or stopping it."


@dataclass(frozen=True)
class Entry:
    key: str                 # canonical lowercase key
    type: str
    title: str
    definition: str          # "" = unknown (only for name-only / pattern entries)
    extra: str = ""
    cs: bool = False         # case-sensitive surface (abbreviations)
    source: str = "glossary"  # glossary | name-list | pattern

    def public(self, surfaces=()):
        d = asdict(self)
        d["type_label"] = TYPE_LABELS.get(self.type, "Medical term")
        d["surfaces"] = sorted(set(surfaces))
        return d


_BY_SURFACE: dict[str, Entry] = {}      # lowercase surface -> entry   (case-insensitive entries)
_BY_SURFACE_CS: dict[str, Entry] = {}   # exact surface -> entry       (abbreviations)


def _add(keys, type_, definition, extra="", title=None, cs=False):
    keys = [keys] if isinstance(keys, str) else list(keys)
    key = keys[0] if cs else keys[0].lower()
    e = Entry(key=key, type=type_, title=title or (keys[0] if cs else keys[0].capitalize()),
              definition=definition, extra=extra, cs=cs)
    for k in keys:
        (_BY_SURFACE_CS if cs else _BY_SURFACE)[k if cs else k.lower()] = e


# ---------------------------------------------------------------------------------------------------- body parts
_B = lambda keys, d, x="": _add(keys, "body", d, x)
_B(["heart"], "The muscle in your chest that pumps blood around your body.", "It beats about 60 to 100 times a minute at rest.")
_B(["lung", "lungs"], "The two organs in your chest that you breathe with. They take in oxygen and remove carbon dioxide.")
_B(["liver"], "A large organ on the right side of your upper belly. It cleans your blood and breaks down many medicines.", "This is why some medicines need liver checks.")
_B(["kidney", "kidneys"], "Two bean-shaped organs in your back that filter waste and extra water out of your blood to make urine.", "Many medicines are cleared from the body by the kidneys.")
_B(["stomach"], "The pouch in your upper belly where swallowed food is mixed with acid and starts to be digested.")
_B(["intestine", "intestines", "bowel", "bowels"], "The long tube after the stomach that absorbs nutrients and carries waste out of the body.")
_B(["colon"], "The large bowel. It takes water out of waste and stores it before you pass it.")
_B(["rectum"], "The last part of the bowel, just before the anus, where stool is held before it leaves the body.")
_B(["bladder"], "The bag in your lower belly that holds urine until you go to the toilet.")
_B(["brain"], "The organ inside your head that controls thinking, memory, movement, feeling and body functions.")
_B(["spine", "spinal cord"], "The column of bones in your back that protects the nerves running down to the rest of your body.")
_B(["skin"], "The outer covering of your body. It protects you from germs, and helps control temperature.")
_B(["artery", "arteries"], "A blood vessel that carries blood from the heart out to the body.", "Blood pressure is the force of blood against artery walls.")
_B(["vein", "veins"], "A blood vessel that carries blood back to the heart.", "Medicines and fluids are often given through a vein (\"intravenously\").")
_B(["blood vessel", "blood vessels"], "Any of the tubes (arteries, veins, small capillaries) that carry blood around the body.")
_B(["aorta"], "The body's main artery. It carries blood from the heart to the rest of the body.")
_B(["esophagus", "oesophagus"], "The tube that carries food and drink from your throat to your stomach.")
_B(["pancreas"], "An organ behind the stomach. It makes insulin, which controls blood sugar, and juices that digest food.")
_B(["gallbladder"], "A small pouch under the liver that stores bile, a fluid that helps digest fat.")
_B(["thyroid"], "A butterfly-shaped gland in the front of the neck. It makes hormones that control energy use and heart rate.")
_B(["spleen"], "An organ on the left side of the belly that filters old blood cells and helps fight infection.")
_B(["throat"], "The passage at the back of your mouth that leads to the stomach (food) and lungs (air).")
_B(["trachea", "windpipe"], "The tube that carries air from your throat down to your lungs.")
_B(["diaphragm"], "The large muscle under the lungs that moves down and up to help you breathe.")
_B(["chest"], "The front part of the body between the neck and the belly. It holds the heart and lungs.")
_B(["abdomen"], "The belly area between the chest and the hips. It holds the stomach, liver, bowels and other organs.")
_B(["pelvis"], "The ring of bones at the base of the spine and between the hips. It supports the bladder and bowel.")
_B(["joint", "joints"], "A place where two bones meet, such as the knee or elbow.")
_B(["muscle", "muscles"], "Tissue that tightens and relaxes to move your body.")
_B(["bone", "bones"], "The hard parts of the skeleton that support the body and protect organs.")
_B(["nerve", "nerves"], "A bundle of fibres that carries signals between the brain and the rest of the body.")
_B(["tendon", "tendons"], "A strong cord that attaches a muscle to a bone.")
_B(["ligament", "ligaments"], "A strong band that connects one bone to another and keeps a joint stable.")
_B(["lymph node", "lymph nodes"], "A small bean-shaped gland that filters germs from body fluid. They can swell during an infection.")
_B(["prostate"], "A small gland below the bladder in men. It makes part of the fluid in semen.")
_B(["uterus", "womb"], "The organ in the pelvis where a baby grows during pregnancy.")
_B(["ovary", "ovaries"], "One of the two organs that store eggs and make female hormones.")
_B(["cervix"], "The narrow lower end of the uterus that opens into the vagina.")
_B(["appendix"], "A small tube attached to the large bowel on the lower right side of the belly.")
_B(["sinus", "sinuses"], "Air-filled spaces in the bones around the nose and eyes.")
_B(["eye", "eyes"], "The organ you see with.")
_B(["ear", "ears"], "The organ you hear with. It also helps with balance.")
_B(["adrenal gland", "adrenal glands"], "A small gland on top of each kidney. It makes hormones such as adrenaline and cortisol.")
_B(["pituitary", "pituitary gland"], "A pea-sized gland under the brain that controls many other hormone glands.")

# ---------------------------------------------------------------------------------------------------- conditions
_C = lambda keys, d, x="": _add(keys, "condition", d, x)
_C(["hypertension", "high blood pressure"], "Blood pushes too hard against the walls of your arteries, all the time.", "It often has no symptoms, so it is found by measuring blood pressure. Over years it can harm the heart, brain and kidneys.")
_C(["hypotension", "low blood pressure"], "Blood pressure that is lower than normal.", "It can cause dizziness or fainting, especially when standing up.")
_C(["blood pressure"], "The force of blood pushing against the walls of your arteries. It is written as two numbers, for example 120/80.", "The top number is when the heart beats; the bottom is when it rests.")
_C(["myocardial infarction", "heart attack"], "Part of the heart muscle is damaged because its blood supply was suddenly blocked.", "Chest pain, shortness of breath or pain spreading to the arm or jaw needs emergency help right away.")
_C(["hypertensive urgency"], "Very high blood pressure without signs of organ damage yet. It needs treatment soon.")
_C(["hypertensive crisis", "hypertensive emergency"], "Dangerously high blood pressure. In an emergency it is harming organs such as the heart, brain or kidneys.", "It needs urgent medical care.")
_C(["diabetic ketoacidosis"], "A serious problem of diabetes. Without enough insulin the body makes too many acids (ketones) and the blood becomes acidic.", "Needs urgent hospital care.")
_C(["dyspnea", "dyspnoea", "shortness of breath"], "Trouble breathing or feeling that you cannot get enough air.", "Sudden or severe shortness of breath needs urgent medical help.")
_C(["edema", "oedema"], "Swelling caused by extra fluid trapped in the body's tissues, often in the feet, ankles or legs.")
_C(["pyrexia", "fever"], "A body temperature higher than normal, usually above about 38 °C (100.4 °F).", "Often a sign that the body is fighting an infection.")
_C(["tachycardia"], "A resting heartbeat that is faster than normal (usually over 100 beats a minute).")
_C(["bradycardia"], "A resting heartbeat that is slower than normal (usually under 60 beats a minute).")
_C(["emesis", "vomiting"], "Throwing up the contents of the stomach.")
_C(["nausea"], "The feeling that you are about to throw up.")
_C(["syncope"], "Fainting: a short loss of consciousness, usually because less blood reaches the brain for a moment.")
_C(["pruritus"], "Itching of the skin.")
_C(["erythema"], "Redness of the skin, caused by more blood flowing in the small vessels.")
_C(["cephalalgia"], "The medical word for a headache.")
_C(["hyperglycemia", "hyperglycaemia", "high blood sugar"], "Too much sugar (glucose) in the blood.", "Common in diabetes. Signs can include thirst, needing to urinate often and tiredness.")
_C(["hypoglycemia", "hypoglycaemia", "low blood sugar"], "Too little sugar (glucose) in the blood.", "Can cause shakiness, sweating, confusion or fainting. It is treated by taking sugar quickly.")
_C(["hyperlipidemia", "high cholesterol"], "Too much fat (such as cholesterol) in the blood.", "Over time it can build up in arteries and raise the risk of heart attack and stroke.")
_C(["cholesterol"], "A waxy fat in the blood. The body needs some, but too much can clog arteries.")
_C(["diabetes"], "A long-term condition where the body cannot keep blood sugar at a healthy level, because of too little insulin or because insulin does not work well.")
_C(["asthma"], "A long-term condition where the airways in the lungs swell and narrow, causing wheezing, coughing and breathlessness.")
_C(["epilepsy"], "A condition of the brain that causes repeated seizures.")
_C(["anemia", "anaemia"], "Too few red blood cells, or too little haemoglobin, so the blood carries less oxygen.", "Often causes tiredness and looking pale.")
_C(["stroke"], "Part of the brain is damaged because its blood supply is blocked or a blood vessel bursts.", "Sudden weakness of the face or arm, or trouble speaking, needs emergency help immediately.")
_C(["renal impairment", "kidney disease", "renal failure", "kidney failure"], "The kidneys are not filtering waste from the blood as well as they should.", "Doses of some medicines must be changed when kidney function is low.")
_C(["chest pain"], "Pain, pressure or tightness in the chest.", "It can come from the heart, lungs, muscles or stomach. New or severe chest pain needs urgent assessment.")
_C(["dizziness"], "Feeling light-headed, unsteady or as if the room is spinning.")
_C(["contraindicated"], "A medicine or treatment that should not be used in this situation because it could cause harm.")
_C(["prognosis"], "The expected course and outcome of an illness.")
_C(["etiology", "aetiology"], "The cause of a disease.")
_C(["exacerbation"], "A flare-up: symptoms of a long-term illness suddenly get worse.")

# ---------------------------------------------------------------------------------------------------- medicine types
_K = lambda keys, d, x="": _add(keys, "class", d, x)
_K(["anticoagulant", "anticoagulants"], "A \"blood thinner\": a medicine that makes blood clot more slowly.", "It does not really thin the blood. It lowers the risk of harmful clots but can make bleeding easier.")
_K(["antiplatelet"], "A medicine that stops blood cells called platelets from sticking together, so clots form less easily.")
_K(["analgesic", "analgesics"], "A pain reliever.")
_K(["antipyretic"], "A medicine that lowers fever.")
_K(["diuretic", "diuretics"], "A \"water pill\": it makes the kidneys remove extra salt and water, so you urinate more.")
_K(["antihypertensive", "antihypertensives"], "A medicine that lowers high blood pressure.")
_K(["ace inhibitor", "ace inhibitors"], "A type of blood pressure and heart medicine that relaxes blood vessels. Names usually end in \"-pril\".", "A dry cough is a common side effect.")
_K(["beta blocker", "beta-blocker", "beta blockers", "beta-blockers"], "A type of heart medicine that slows the heartbeat and lowers blood pressure. Names usually end in \"-olol\".", "Do not stop suddenly unless told to.")
_K(["statin", "statins"], "A type of medicine that lowers cholesterol. Names usually end in \"-statin\".")
_K(["antibiotic", "antibiotics"], "A medicine that kills bacteria or stops them growing. It does not work against viruses such as colds and flu.")
_K(["nsaid", "nsaids"], "Non-steroidal anti-inflammatory drug, such as ibuprofen: it eases pain, swelling and fever.", "It can irritate the stomach and strain the kidneys.")
_K(["opioid", "opioids"], "A strong type of pain medicine, such as morphine or oxycodone.", "It can cause drowsiness and constipation, and carries a risk of dependence.")
_K(["corticosteroid", "corticosteroids", "steroid", "steroids"], "A medicine that reduces swelling and calms the immune system. It is not the same as muscle-building steroids.")

# ---------------------------------------------------------------------------------------------------- abbreviations
_A = lambda k, d, x="": _add([k], "abbr", d, x, title=k, cs=True)
_A("BID", "Twice a day.", "From the Latin \"bis in die\".")
_A("TID", "Three times a day.")
_A("QID", "Four times a day.")
_A("PRN", "Only when needed.", "For example, for pain, as you feel you need it.")
_A("QHS", "At bedtime.")
_A("PO", "By mouth (swallowed).")
_A("IV", "Into a vein (intravenous).")
_A("mmHg", "Millimetres of mercury: the unit blood pressure is measured in.")

# ---------------------------------------------------------------------------------------------------- medicines
_M = lambda keys, d, x="": _add(keys, "medicine", d, (x + " " if x else "") + DRUG_TIP)
_M(["warfarin"], "A blood thinner. It lowers the risk of harmful blood clots.", "Needs regular blood tests (INR). Report unusual bleeding or bruising. Keep your intake of leafy green vegetables steady.")
_M(["heparin"], "A fast-acting blood thinner given by injection or into a vein. It prevents and treats blood clots.")
_M(["apixaban", "rivaroxaban"], "A blood thinner (anticoagulant). It lowers the risk of clots and stroke, for example with an irregular heartbeat (atrial fibrillation).", "Do not skip doses. Report unusual bleeding.")
_M(["clopidogrel"], "A medicine that stops platelets sticking together. It lowers the risk of clots, heart attack and stroke.", "You may bruise or bleed more easily.")
_M(["aspirin"], "A medicine for pain, fever and swelling. In low doses it helps prevent clots.", "It can irritate the stomach and cause bleeding. Not for children with fever.")
_M(["lisinopril"], "A blood pressure and heart medicine (ACE inhibitor). It relaxes blood vessels.", "A dry cough is common. Swelling of the face, lips or tongue needs urgent help. Not for use in pregnancy.")
_M(["losartan"], "A blood pressure medicine (ARB). It relaxes blood vessels and can protect the kidneys in diabetes.", "Not for use in pregnancy.")
_M(["amlodipine"], "A blood pressure and chest pain (angina) medicine. It relaxes blood vessels.", "Ankle swelling is a common side effect.")
_M(["metoprolol", "carvedilol", "bisoprolol"], "A beta blocker: it slows the heartbeat and lowers blood pressure. Also used for heart failure and chest pain.", "Do not stop suddenly unless told to.")
_M(["propranolol"], "A beta blocker used for blood pressure, heart rhythm, migraine prevention and tremor.", "Do not stop suddenly unless told to.")
_M(["furosemide"], "A strong water pill. It removes extra fluid, for example in heart failure or swelling.", "You will urinate more. It can lower potassium.")
_M(["hydrochlorothiazide"], "A mild water pill used mainly to lower blood pressure.")
_M(["spironolactone"], "A water pill that keeps potassium in the body. Used for heart failure, blood pressure and fluid build-up.", "Blood tests check potassium.")
_M(["digoxin"], "A heart medicine for heart failure and some irregular heartbeats. It strengthens the heartbeat and slows it.", "The safe dose range is narrow. Report nausea, vomiting or vision changes.")
_M(["amiodarone"], "A medicine that keeps the heart beating in a regular rhythm.", "It can affect the thyroid, lungs and liver, so regular checks are needed.")
_M(["atorvastatin", "simvastatin"], "A statin: it lowers cholesterol and the risk of heart attack and stroke.", "Report unexplained muscle pain or weakness.")
_M(["metformin"], "A medicine for type 2 diabetes. It lowers blood sugar.", "Usually taken with meals. Stomach upset is common at first.")
_M(["insulin"], "A hormone medicine that lowers blood sugar. Usually injected.", "Too much can cause low blood sugar.")
_M(["acetaminophen", "paracetamol"], "A pain reliever and fever reducer.", "Too much can seriously damage the liver. Check other products you take for the same ingredient.")
_M(["ibuprofen", "naproxen"], "An anti-inflammatory pain reliever (NSAID). It eases pain, swelling and fever.", "Take with food. It can irritate the stomach and strain the kidneys.")
_M(["prednisone"], "A steroid that reduces swelling and calms the immune system.", "After long use, do not stop suddenly. It can raise blood sugar and affect mood and sleep.")
_M(["amoxicillin"], "A penicillin-type antibiotic for bacterial infections.", "Tell your clinician if you are allergic to penicillin.")
_M(["azithromycin"], "An antibiotic for bacterial infections, such as chest and throat infections.")
_M(["ciprofloxacin"], "An antibiotic for bacterial infections, such as urinary infections.", "Stop it and call your clinician if you get tendon pain or swelling.")
_M(["levothyroxine"], "A replacement for the thyroid hormone, for an underactive thyroid.", "Usually taken on an empty stomach, in the morning.")
_M(["omeprazole"], "A medicine that lowers stomach acid. Used for heartburn and stomach ulcers.")
_M(["gabapentin"], "A medicine for nerve pain and some types of seizures.", "It can cause sleepiness and dizziness.")
_M(["tramadol"], "A pain reliever that acts like an opioid.", "It can cause drowsiness. Avoid alcohol. Can lead to dependence.")
_M(["oxycodone", "morphine", "fentanyl", "hydrocodone"], "A strong opioid pain reliever.", "It can cause drowsiness, constipation and slow breathing. Avoid alcohol. Risk of dependence. Keep it away from other people.")
_M(["lorazepam", "diazepam", "alprazolam"], "A benzodiazepine: a calming medicine used for anxiety, seizures or sleep problems.", "It causes drowsiness. Avoid alcohol. Can lead to dependence.")
_M(["sertraline", "fluoxetine", "citalopram", "escitalopram"], "An SSRI antidepressant. It is used for depression and anxiety.", "It can take several weeks to work. Do not stop suddenly.")
_M(["lithium"], "A mood-stabilising medicine, mainly for bipolar disorder.", "The safe dose range is narrow, so blood levels are checked. Stay well hydrated.")
_M(["carbamazepine", "phenytoin", "levetiracetam"], "A medicine to prevent seizures.", "Do not stop suddenly. Blood levels may be checked.")
_M(["albuterol", "salbutamol"], "A quick-relief inhaler. It opens the airways in asthma and similar lung conditions.", "If you need it more often than usual, tell your clinician.")
_M(["methotrexate"], "A medicine for some cancers and for immune conditions such as rheumatoid arthritis.", "For arthritis it is usually taken only once a week, so check the schedule carefully. Regular blood tests are needed.")
_M(["tacrolimus"], "A medicine that stops the body rejecting a transplanted organ.", "Blood levels are checked regularly.")

# ---------------------------------------------------------------------------------------------------- optional user file
def _load_extra():
    p = os.environ.get("SIPG_GLOSSARY")
    if p and os.path.exists(p):
        for r in json.load(open(p, encoding="utf8")):
            _add(r["keys"], r.get("type", "other"), r["definition"], r.get("extra", ""), r.get("title"), r.get("cs", False))


_load_extra()

# ---------------------------------------------------------------------------------------------------- finders
_NAME_ONLY = {n: Entry(n, "medicine", n.capitalize(), "", "", False, "name-list")
              for n in CRITICAL_TERMS if n not in _BY_SURFACE}

_SUFFIX = (r"itis|aemia|emia|osis|pathy|ectomy|otomy|ostomy|plasty|oscopy|algia|megaly|penia|plegia|"
           r"cillin|mycin|pril|sartan|olol|statin|azole|prazole|tidine|dipine|floxacin|gliptin|gliflozin|"
           r"glutide|triptan|oxetine|setron|zepam|zolam|mab")
_STOP = {"analysis", "basis", "emphasis", "synthesis", "hypothesis", "thesis", "genesis", "oasis", "crisis", "diagnosis",
         "academia", "sympathy", "empathy", "telepathy", "apathy", "antipathy", "osteopathy", "homeopathy", "prognosis",
         "mab"}
_PATTERN = re.compile(rf"(?<![\w-])([A-Za-z]{{3,}}(?:{_SUFFIX}))(?![\w-])", re.I)


def _alt(words):
    return "|".join(re.escape(w) for w in sorted(words, key=lambda w: (-len(w), w)))


_CI_WORDS = list(_BY_SURFACE) + list(_NAME_ONLY)
_MAIN = re.compile(rf"(?<![\w-])(?:(?i:{_alt(_CI_WORDS)})|{_alt(_BY_SURFACE_CS)})(?![\w-])")


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    surface: str
    entry: Entry


def lookup(term: str):
    t = term.strip()
    return _BY_SURFACE_CS.get(t) or _BY_SURFACE.get(t.lower()) or _NAME_ONLY.get(t.lower())


def find_terms(text: str, guess: bool = True) -> list:
    """Non-overlapping matches, left to right. guess=False disables the word-shape guesses."""
    out, taken = [], []
    for m in _MAIN.finditer(text):
        e = lookup(m.group(0))
        if e:
            out.append(Match(m.start(), m.end(), m.group(0), e)); taken.append((m.start(), m.end()))
    if guess:
        for m in _PATTERN.finditer(text):
            w = m.group(1)
            if w.lower() in _STOP or lookup(w) or any(a < m.end() and m.start() < b for a, b in taken):
                continue
            out.append(Match(m.start(), m.end(), w, Entry(w.lower(), "other", w.capitalize(), "", "", False, "pattern")))
    return sorted(out, key=lambda x: x.start)


def collect(text: str, guess: bool = True) -> list:
    """Unique entries found in `text`, as JSON-ready dicts with the surface forms seen (used by the dashboard)."""
    seen: dict = {}
    for m in find_terms(text, guess):
        e, surf = seen.setdefault(m.entry.key + "|" + m.entry.source, (m.entry, set()))
        surf.add(m.surface)
    return [e.public(s) for e, s in seen.values()]
