Import os
import sys
import queue
import sounddevice as sd
import json
import subprocess
import datetime
import time
import fasttext
from vosk import Model, KaldiRecognizer
from difflib import SequenceMatcher

#--------------------------------------------
from RPLCD.i2c import CharLCD
lcd = CharLCD('PCF8574', 0x27)   # change 0x27 if your address is different
def lcd_show(line1="", line2=""):
    lcd.clear()
    lcd.write_string(line1[:16])
    lcd.cursor_pos = (1, 0)
    lcd.write_string(line2[:16])
#------------------------------------------

# ===========================
# 1. CONFIGURATION
# ===========================
MODEL_PATH = "model"
MODEL_BIN = "brain_model.ftz"  # <--- Loads your  trained brain
PERSONAL_FILE = "personal_memory.json"
SAMPLE_RATE = 44100
INPUT_DEVICE = 1
WAKE_WORDS = ["bharat", "भारत", "barat", "bart", "varat", "parrot", "birth", "baarat"]
WAKE_WINDOW = 15

device_state = {"light": "off", "fan": "off"}
is_awake = False
last_interaction_time = 0

# ===========================
# 2. MEMORY LAYER
# ===========================
def load_personal_memory():
    if not os.path.exists(PERSONAL_FILE):
        with open(PERSONAL_FILE, 'w', encoding='utf-8') as f: json.dump({"name": None}, f)
    with open(PERSONAL_FILE, 'r', encoding='utf-8') as f: return json.load(f)

def save_personal_memory(data):
    with open(PERSONAL_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)

user_memory = load_personal_memory()

# ===========================
# 3. LOAD EXISTING BRAIN
# ===========================
print(f"🧠 Loading Brain from {MODEL_BIN}...")

if not os.path.exists(MODEL_BIN):
    print(f"❌ ERROR: '{MODEL_BIN}' not found!")
    print("👉 Please copy the 'brain_model.bin' file from your Computer to this folder.")
    sys.exit(1)

try:
    # LOAD THE MODEL (No Training)
    ai_brain = fasttext.load_model(MODEL_BIN)
    print("✅ Brain Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

def get_intent(text):
    if not ai_brain: return "unknown", 0.0
    try:
        prediction = ai_brain.predict(text)
        return prediction[0][0].replace("__label__", ""), prediction[1][0]
    except:
        return "unknown", 0.0

# ===========================
# 4. HELPER FUNCTIONS
# ===========================
def speak(text):
    print(f"🗣 Assistant: {text}")
    try:
        subprocess.run(['espeak-ng', '-v', 'hi', '-s', '140', text], shell=False)
    except:
        pass

def hindi_to_english_char(text):
    mapping = {'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
               'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ट': 't',
               'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n', 'प': 'p',
               'फ': 'f', 'ब': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh',
               'स': 's', 'ह': 'h', 'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'े': 'e', 'ै': 'ai', 'ो': 'o',
               'ौ': 'au', '्': '', 'ं': 'n', 'ः': 'h'}
    result = []
    for char in text:
        if char in mapping: result.append(mapping[char].upper())
        elif char.isalnum(): result.append(char.upper())
    return result

def spell_out(text):
    if not text: return ""
    chunks = hindi_to_english_char(text)
    return "".join(chunks)

# --- FUZZY LOGIC ENGINE ---
def is_similar(word, target_list, threshold=0.8):
    if not word: return False
    if word in target_list: return True
    for target in target_list:
        if SequenceMatcher(None, word, target).ratio() >= threshold: return True
    return False

def contains_fuzzy(text, target_list, threshold=0.75):
    words = text.split()
    for w in words:
        if is_similar(w, target_list, threshold): return True
    return False

def extract_name(text):
    triggers = ["मेरा नाम", "बुलाओ", "पुकारो"]
    used_trigger = None
    for t in triggers:
        if t in text: used_trigger = t; break
    if not used_trigger: return None
    parts = text.split(used_trigger)
    if len(parts) < 2: return None
    raw_name = parts[1]
    for r in [" है", " हैं", " था", " को", " का"]: raw_name = raw_name.replace(r, "")
    return raw_name.strip()

# ===========================
# 5. FULL MATH ENGINE (0-100)
# ===========================
def perform_math(text):
    word_map = {
        'शून्य': 0, 'जीरो': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पाँच': 5, 'पांच': 5, 'छह': 6, 'चे': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
        'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15, 'सोलह': 16, 'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19, 'बीस': 20,
        'इक्कीस': 21, 'बाइस': 22, 'तेइस': 23, 'चौबीस': 24, 'पच्चीस': 25, 'छब्बीस': 26, 'सत्ताइस': 27, 'अट्ठाइस': 28, 'उनतीस': 29, 'तीस': 30,
        'इकतीस': 31, 'बत्तीस': 32, 'तैंतीस': 33, 'चौंतीस': 34, 'पैंतीस': 35, 'छत्तीस': 36, 'सैंतीस': 37, 'अड़तीस': 38, 'उनतालीस': 39, 'चालीस': 40,
        'इकतालीस': 41, 'बयालीस': 42, 'तैंतालीस': 43, 'चवालीस': 44, 'पैंतालीस': 45, 'छियालीस': 46, 'सैंतालीस': 47, 'अड़तालीस': 48, 'उनचास': 49, 'पचास': 50,
        'इक्यावन': 51, 'बावन': 52, 'तिरेपन': 53, 'चौवन': 54, 'पचपन': 55, 'छप्पन': 56, 'सत्तावन': 57, 'अट्ठावन': 58, 'उनसठ': 59, 'साठ': 60,
        'इकसठ': 61, 'बासठ': 62, 'तिरसठ': 63, 'चौंसठ': 64, 'पैंसठ': 65, 'सियासठ': 66, 'सड़सठ': 67, 'अड़सठ': 68, 'उनहत्तर': 69, 'सत्तर': 70,
        'इकहत्तर': 71, 'बहत्तर': 72, 'तिहत्तर': 73, 'चौहत्तर': 74, 'पचहत्तर': 75, 'छिहत्तर': 76, 'सतहत्तर': 77, 'अठहत्तर': 78, 'उन्यासी': 79, 'अस्सी': 80,
        'इक्यासी': 81, 'बयासी': 82, 'तिरासी': 83, 'चौरासी': 84, 'पचासी': 85, 'छियासी': 86, 'सत्तासी': 87, 'अट्ठासी': 88, 'नवासी': 89, 'नब्बे': 90,
        'इक्यानवे': 91, 'बानवे': 92, 'तिरानवे': 93, 'चौरानवे': 94, 'पचानवे': 95, 'छियानवे': 96, 'सत्तानवे': 97, 'अट्ठानवे': 98, 'निन्यानवे': 99, 'सौ': 100,
        'हजार': 1000, 'लाख': 100000, 'करोड़': 10000000,
        'वन': 1, 'टू': 2, 'थ्री': 3, 'फोर': 4, 'फाइव': 5, 'फाई': 5, 'सिक्स': 6, 'सेवन': 7, 'एट': 8, 'नाइन': 9, 'टेन': 10
    }
    nums = []
    tokens = text.split()

    for t in tokens:
        if t.isdigit():
            nums.append(int(t))
        elif t in word_map:
            nums.append(word_map[t])
        else:
            # Fuzzy match
            for key, val in word_map.items():
                if isinstance(val, int) and SequenceMatcher(None, t, key).ratio() > 0.85:
                    nums.append(val)
                    break

    if len(nums) < 2:
        return None

    if contains_fuzzy(text, ["प्लस", "जोड़ो", "धन"], 0.6):
        spoken = f"{nums[0]} plus {nums[1]} hota hai {nums[0] + nums[1]}"
        lcd = f"{nums[0]}+{nums[1]}={nums[0] + nums[1]}"
        return spoken, lcd

    elif contains_fuzzy(text, ["माइनस", "घटाओ", "कम"], 0.6):
        spoken = f"{nums[0]} minus {nums[1]} hota hai {nums[0] - nums[1]}"
        lcd = f"{nums[0]}-{nums[1]}={nums[0] - nums[1]}"
        return spoken, lcd

    elif contains_fuzzy(text, ["भाग", "डिवाइड"], 0.6):
        if nums[1] == 0:
            return "Zero se divide nahi kar sakte.", "ERROR"

        spoken = f"{nums[0]} divide by {nums[1]} hota hai {nums[0] / nums[1]:.1f}"
        lcd = f"{nums[0]}/{nums[1]}={nums[0] / nums[1]:.1f}"
        return spoken, lcd

    return None


def check_hardcoded_intent(text):
    if is_similar("मेरा", text.split(), 0.8) and is_similar("नाम", text.split(), 0.8):
        if any(q in text for q in ["क्या", "बताओ", "किया"]): return "ask_name"
    if contains_fuzzy(text, ["तुम", "कौन", "किसने", "बनाया"]): return "ask_identity"
    if contains_fuzzy(text, ["रुक", "स्टॉप", "बस", "बंद"]): return "stop"
    if contains_fuzzy(text, ["लाइट", "बत्ती", "बल्ब"]):
        if contains_fuzzy(text, ["ऑन", "चालू", "जलाओ"]): return "light_on"
        if contains_fuzzy(text, ["ऑफ", "बंद", "बुझाओ"]): return "light_off"
    if contains_fuzzy(text, ["फैन", "पंखा", "हवा"]):
        if contains_fuzzy(text, ["ऑन", "चालू", "चलाओ"]): return "fan_on"
        if contains_fuzzy(text, ["ऑफ", "बंद", "रोको"]): return "fan_off"
    if contains_fuzzy(text, ["टाइम", "समय", "बजे"]): return "time"
    if contains_fuzzy(text, ["डेट", "तारीख", "दिन"]): return "date"
    if contains_fuzzy(text, ["प्लस", "माइनस", "जोड़ो", "घटाओ", "भाग", "डिवाइड"]): return "math"
    return None

def execute_intent(intent, text=""):
    global user_memory, device_state

    if intent == "date":
        today = datetime.datetime.now()
        target_date = today
        if contains_fuzzy(text, ["कल"]):
            target_date += datetime.timedelta(days=1)
            day_str = "Kal"
        elif contains_fuzzy(text, ["परसों"]):
            target_date += datetime.timedelta(days=2)
            day_str = "Parson"
        else:
            day_str = "Aaj"
        date_str = target_date.strftime("%d %B %Y")
        lcd_show("DATE", date_str)   # 👈 ADD THIS LINE HERE
        speak(f"{day_str} ki tarikh {date_str} hai.")

    elif intent == "math":
        result = perform_math(text)

        if result:
            spoken_text, lcd_text = result
            lcd_show("CALCULATION", lcd_text)
            speak(spoken_text)
        else:
            speak("Maaf kijiye, number samajh nahi aaya.")



    elif intent == "light_on":
        if device_state["light"] == "on":
            lcd_show("LIGHT", "ALREADY ON")
            speak("Light pehle se on hai.")
        else:
            device_state["light"] = "on"
            lcd_show("LIGHT", "ON")
            speak("Light on kar di hai.")

    elif intent == "light_off":
        if device_state["light"] == "off":
            lcd_show("LIGHT", "ALREADY OFF")
            speak("Light pehle se off hai.")
        else:
            device_state["light"] = "off"
            lcd_show("LIGHT", "OFF")
            speak("Light off kar di hai.")

    elif intent == "fan_on":
        if device_state["fan"] == "on":
            lcd_show("FAN", "ALREADY ON")
            speak("Fan pehle se chalu hai.")
        else:
            device_state["fan"] = "on"
            lcd_show("FAN", "ON")
            speak("Fan chalu kar diya hai.")

    elif intent == "fan_off":
        if device_state["fan"] == "off":
            lcd_show("FAN", "ALREADY OFF")
            speak("Fan pehle se band hai.")
        else:
            device_state["fan"] = "off"
            lcd_show("FAN", "OFF")
            speak("Fan band kar diya hai.")


    elif intent == "stop":
        lcd_show("THANK YOU", "")   # 👈 FIRST
        speak("Alvida.")
        time.sleep(2)               # 👈 allow user to see it
        sys.exit(0)

    elif intent == "greet":
        lcd_show("NAMASTE", "")     # 👈 ADD
        speak("Namaste!")

    elif intent == "time":
        now = datetime.datetime.now().strftime("%I:%M %p")
        lcd_show("TIME", now)       # 👈 ADD
        speak(f"Abhi {now} bajey hain.")

    elif intent == "ask_name":
        if user_memory["name"]:
            name = user_memory['name']
            spelling = spell_out(name)
            lcd_show("YOUR NAME", name)   # 👈 ADD
            speak(f"Aapka naam {name} hai. {spelling}")
        else:
            lcd_show("NAME", "NOT SET")   # 👈 ADD
            speak("Mujhe aapka naam nahi pata.")

    elif intent == "ask_identity":
        lcd_show("I AM", "BHARAT SOC")    # 👈 ADD
        speak("Main Bharat SOC hoon.")


# ===========================
# 6. MAIN LOOP
# ===========================
def process_command(text):
    global is_awake, last_interaction_time, user_memory
    print(f"👂 Raw: {text}")

    wake_word_heard = False
    for w in WAKE_WORDS:
        if w in text: wake_word_heard = True; text = text.replace(w, "").strip(); break

    current_time = time.time()
    if wake_word_heard: is_awake = True; last_interaction_time = current_time;
    elif is_awake:
        if (current_time - last_interaction_time) > WAKE_WINDOW: is_awake = False; return
        last_interaction_time = current_time
    else: return

    if not text:
        if wake_word_heard: speak("Ji?"); return

    # 1. SAFETY NET
    intent = check_hardcoded_intent(text)
    if intent:
        print(f"🛡️ Safety Net: {intent}")
        execute_intent(intent, text)
        return

    # 2. NAME LEARNING
    if "मेरा नाम" in text:
        if not any(q in text for q in ["क्या", "बताओ", "किया"]):
            name = extract_name(text)
            if name:
                user_memory["name"] = name
                save_personal_memory(user_memory)
                spelling = spell_out(name)
                speak(f"Theek hai {name}, maine yaad kar liya. {spelling}")
                return

    # 3. AI BRAIN
    intent, confidence = get_intent(text)
    print(f"🔮 AI: {intent} ({confidence:.2f})")

    if confidence > 0.4:
        execute_intent(intent, text)
    else:
        print("🤫 Ignoring low confidence...")

# ===========================
# 7. INIT
# ===========================
if not os.path.exists(MODEL_PATH): print("❌ Error: Model missing."); sys.exit(1)
model = Model(MODEL_PATH)
rec = KaldiRecognizer(model, SAMPLE_RATE)
rec.SetWords(True)
rec.SetPartialWords(True)
q = queue.Queue()

def callback(indata, frames, time, status): q.put(bytes(indata))

print("\n🟢 BharatSOC (Run-Only Edition) Ready.")
speak("System Online.")
lcd_show("SYSTEM", "ONLINE")   # 👈 ADD EXACTLY HERE

try:
    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=4000, device=INPUT_DEVICE, dtype='int16', channels=1,latency='low', callback=callback):
        while True:
            data = q.get()
            lcd_show("SPEAK", "")
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    lcd.clear()    
                    process_command(text)
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
