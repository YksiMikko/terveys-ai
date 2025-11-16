import streamlit as st
import google.generativeai as genai
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings

st.set_page_config(page_title="Älykäs Terveys-AI", layout="wide")
st.title("💡 Älykäs Oireanalysoija")

warnings.filterwarnings('ignore')

# --- API-AVAINTEN LATAUS ---
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    WHO_CLIENT_ID = st.secrets["WHO_CLIENT_ID"]
    WHO_CLIENT_SECRET = st.secrets["WHO_CLIENT_SECRET"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError as e:
    st.error(f"⚠️ Avain '{e.args[0]}' puuttuu! Lisää se Secrets-asetuksiin.")
    st.stop()

# --- FUNKTIOT ---

@st.cache_resource
def luo_ja_kouluta_malli():
    """Luo ja kouluttaa mallin (ajetaan vain kerran)."""
    data = {
        'fever': [], 'headache': [], 'cough': [], 'sore throat': [], 
        'shortness of breath': [], 'chest pain': [], 'abdominal pain': [], 
        'nausea': [], 'fatigue': [], 'muscle pain': [], 'vakavuus': []
    }
    np.random.seed(42)
    for _ in range(1000):
        if np.random.random() < 0.5:
            data['fever'].append(np.random.choice([0, 1], p=[0.7, 0.3]))
            data['shortness of breath'].append(0)
            data['headache'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['chest pain'].append(0)
            data['cough'].append(np.random.choice([0, 1], p=[0.6, 0.4]))
            data['sore throat'].append(np.random.choice([0, 1], p=[0.4, 0.6]))
            data['abdominal pain'].append(np.random.choice([0, 1], p=[0.8, 0.2]))
            data['nausea'].append(np.random.choice([0, 1], p=[0.9, 0.1]))
            data['fatigue'].append(np.random.choice([0, 1], p=[0.4, 0.6]))
            data['muscle pain'].append(np.random.choice([0, 1], p=[0.6, 0.4]))
            data['vakavuus'].append(0)
        elif np.random.random() < 0.85:
            data['fever'].append(np.random.choice([0, 1], p=[0.2, 0.8]))
            data['shortness of breath'].append(np.random.choice([0, 1], p=[0.7, 0.3]))
            data['headache'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['chest pain'].append(np.random.choice([0, 1], p=[0.8, 0.2]))
            data['cough'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['sore throat'].append(np.random.choice([0, 1], p=[0.5, 0.5]))
            data['abdominal pain'].append(np.random.choice([0, 1], p=[0.6, 0.4]))
            data['nausea'].append(np.random.choice([0, 1], p=[0.5, 0.5]))
            data['fatigue'].append(np.random.choice([0, 1], p=[0.2, 0.8]))
            data['muscle pain'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['vakavuus'].append(1)
        else:
            data['fever'].append(np.random.choice([0, 1], p=[0.1, 0.9]))
            data['shortness of breath'].append(np.random.choice([0, 1], p=[0.2, 0.8]))
            data['headache'].append(np.random.choice([0, 1], p=[0.2, 0.8]))
            data['chest pain'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['cough'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['sore throat'].append(np.random.choice([0, 1], p=[0.7, 0.3]))
            data['abdominal pain'].append(np.random.choice([0, 1], p=[0.5, 0.5]))
            data['nausea'].append(np.random.choice([0, 1], p=[0.4, 0.6]))
            data['fatigue'].append(np.random.choice([0, 1], p=[0.1, 0.9]))
            data['muscle pain'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['vakavuus'].append(2)
    
    df = pd.DataFrame(data)
    feature_columns = [col for col in df.columns if col != 'vakavuus']
    X = df[feature_columns]
    y = df['vakavuus']
    malli = RandomForestClassifier(n_estimators=100, random_state=42)
    malli.fit(X, y)
    return malli, feature_columns

def hae_bearer_token():
    """Hakee WHO:n Bearer-tokenin."""
    token_url = "https://icdaccessmanagement.who.int/connect/token"
    payload = {
        'grant_type': 'client_credentials',
        'client_id': WHO_CLIENT_ID,
        'client_secret': WHO_CLIENT_SECRET,
        'scope': 'icdapi_access'
    }
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    try:
        response = requests.post(token_url, data=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            st.error(f"Token-virhe: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Token-haku epäonnistui: {e}")
        return None

def hae_who_oiretieto(hakusana_englanniksi, token):
    """Hakee oiretiedon WHO:sta."""
    if not token:
        return {'löytyi': False, 'viesti': 'Token puuttuu'}
    
    search_url = "https://id.who.int/icd/release/11/2024-01/mms/search"
    params = {'q': hakusana_englanniksi}
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
        'Accept-Language': 'en',
        'API-Version': 'v2'
    }
    
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return {'löytyi': False, 'viesti': f"Virhe: {response.status_code}"}
        
        data = response.json()
        if not data.get('destinationEntities'):
            return {'löytyi': False, 'viesti': f"Ei tuloksia"}
        
        entity_url = data['destinationEntities'][0].get('id', '')
        detail_response = requests.get(entity_url, headers=headers, timeout=10)
        
        if detail_response.status_code == 200:
            detail_data = detail_response.json()
            return {
                'löytyi': True,
                'otsikko': detail_data.get('title', {}).get('@value', 'Ei otsikkoa'),
                'koodi': data['destinationEntities'][0].get('theCode', 'Ei koodia'),
                'määritelmä': detail_data.get('definition', {}).get('@value', 'Ei määritelmää')
            }
        return {'löytyi': False, 'viesti': 'Lisätiedot epäonnistuivat'}
    except Exception as e:
        return {'löytyi': False, 'viesti': f'Virhe: {str(e)}'}

def analysoi_oireet(kayttajan_viesti, malli, piirteet, status_placeholder):
    """Orkestroi analyysiprosessin."""
    try:
        status_placeholder.text("1/5: Haetaan WHO-tokenia...")
        bearer_token = hae_bearer_token()
        if not bearer_token:
            return "❌ WHO-autentikointi epäonnistui"

        status_placeholder.text("2/5: Tunnistetaan oireita...")
        gemini_tunnistus = genai.GenerativeModel('gemini-2.0-flash-exp')
        prompt_tunnistus = f"""
        Analyze this text and extract medical symptoms.
        Return ONLY comma-separated list in ENGLISH lowercase.
        Example: "fever,headache"
        Text: "{kayttajan_viesti}"
        """
        vastaus = gemini_tunnistus.generate_content(prompt_tunnistus)
        tunnistetut_oireet_en = [o.strip().lower() for o in vastaus.text.strip().split(',') if o.strip()]
        
        if not tunnistetut_oireet_en:
            return "❌ En tunnistanut oireita"

        status_placeholder.text(f"3/5: Haetaan WHO-tietoja ({len(tunnistetut_oireet_en)} oiretta)...")
        who_tiedot = [hae_who_oiretieto(oire, bearer_token) for oire in tunnistetut_oireet_en]

        status_placeholder.text("4/5: Arvioidaan vakavuutta...")
        syote_vektori = pd.DataFrame([np.zeros(len(piirteet))], columns=piirteet)
        for oire in tunnistetut_oireet_en:
            if oire in piirteet:
                syote_vektori[oire] = 1
        
        ennuste = malli.predict(syote_vektori)[0]
        vakavuus_map = {0: "Lievä", 1: "Keskivakava", 2: "Vakava"}
        vakavuus_teksti = vakavuus_map.get(ennuste, "Tuntematon")

        status_placeholder.text("5/5: Kootaan vastaus...")
        who_teksti_en = "\n".join([
            f"- {t['otsikko']} (Code: {t['koodi']}): {t['määritelmä']}" 
            for t in who_tiedot if t['löytyi']
        ])
        
        prompt_lopputulos = f"""
        You are an empathetic AI health assistant. Create response IN FINNISH.
        IMPORTANT: Remind user you're AI, not a doctor.

        Data:
        - User: "{kayttajan_viesti}"
        - Symptoms: {', '.join(tunnistetut_oireet_en)}
        - Severity: "{vakavuus_teksti}"
        - WHO: {who_teksti_en}
        """
        
        gemini_koonti = genai.GenerativeModel('gemini-2.0-flash-exp')
        lopullinen_vastaus = gemini_koonti.generate_content(prompt_lopputulos)
        status_placeholder.empty()
        return lopullinen_vastaus.text
        
    except Exception as e:
        st.error(f"Virhe: {e}")
        return "❌ Tekninen virhe tapahtui"

# --- KÄYTTÖLIITTYMÄ ---

st.markdown("""
### 📋 Kuvaile oireitasi

**TÄRKEÄÄ:** Tämä on tekninen demonstraatio, ei lääketieteellinen työkalu. 
Hakeudu aina lääkäriin vakavien oireiden kanssa.
""")

vakavuusmalli, piirteet = luo_ja_kouluta_malli()

user_input = st.text_area(
    "Syötä oireesi:", 
    "I feel very tired, I have a strong headache and a bit of a cough.",
    height=100
)

if st.button("🔍 Analysoi oireet", type="primary"):
    if user_input.strip():
        status_placeholder = st.empty()
        with st.spinner('Käsitellään...'):
            tulos = analysoi_oireet(user_input, vakavuusmalli, piirteet, status_placeholder)
            st.markdown("---")
            st.markdown("### 📊 Tulokset:")
            st.markdown(tulos)
    else:
        st.warning("⚠️ Syötä oireesi ensin")
