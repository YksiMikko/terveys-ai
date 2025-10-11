# app.py
import streamlit as st
import google.generativeai as genai
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import json
import warnings

# --- SIVUN ASETUKSET ---
st.set_page_config(page_title="Älykäs Terveys-AI", layout="wide")
st.title("💡 Älykäs Oireanalysoija")
st.markdown("""
Kuvaile oireitasi alla olevaan tekstikenttään. Tekoäly tunnistaa oireet, hakee niille viralliset määritelmät WHO:lta, 
arvioi niiden muodostaman kokonaisuuden vakavuutta ja tuottaa lopuksi selkeän yhteenvedon.

**TÄRKEÄÄ:** Tämä sovellus on vain tekninen demonstraatio eikä korvaa lääketieteen ammattilaisen arviota. 
Hakeudu aina lääkäriin, jos olet huolissasi terveydestäsi.
""")

warnings.filterwarnings('ignore')

# --- API-AVAINTEN JA SALAISUUKSIEN LATAUS ---
# Streamlit Cloudissa nämä lisätään sovelluksen asetuksista.
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    WHO_CLIENT_ID = st.secrets["WHO_CLIENT_ID"]
    WHO_CLIENT_SECRET = st.secrets["WHO_CLIENT_SECRET"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError as e:
    st.error(f"Avainta ei löytynyt! Varmista, että olet lisännyt salaisuuden '{e.args[0]}' Streamlit Cloudin asetuksiin.")
    st.stop()

# --- FUNKTIOT COLABISTA (Päivitetty Streamlitille) ---

# Käytetään välimuistia, jotta mallia ei kouluteta joka kerta uudelleen.
@st.cache_resource
def luo_ja_kouluta_malli():
    # Tämä on täsmälleen sama funktio kuin Colabissa.
    data = {
        'fever': [], 'headache': [], 'cough': [], 'sore throat': [], 
        'shortness of breath': [], 'chest pain': [], 'abdominal pain': [], 
        'nausea': [], 'fatigue': [], 'muscle pain': [], 'vakavuus': []
    }
    np.random.seed(42)
    for _ in range(1000):
        if np.random.random() < 0.5: # Lievä
            data['fever'].append(np.random.choice([0, 1], p=[0.7, 0.3])); data['shortness of breath'].append(0)
            data['headache'].append(np.random.choice([0, 1], p=[0.3, 0.7])); data['chest pain'].append(0)
            data['cough'].append(np.random.choice([0, 1], p=[0.6, 0.4])); data['sore throat'].append(np.random.choice([0, 1], p=[0.4, 0.6]))
            data['abdominal pain'].append(np.random.choice([0, 1], p=[0.8, 0.2])); data['nausea'].append(np.random.choice([0, 1], p=[0.9, 0.1]))
            data['fatigue'].append(np.random.choice([0, 1], p=[0.4, 0.6])); data['muscle pain'].append(np.random.choice([0, 1], p=[0.6, 0.4]))
            data['vakavuus'].append(0)
        elif np.random.random() < 0.85: # Keskivakava
            data['fever'].append(np.random.choice([0, 1], p=[0.2, 0.8])); data['shortness of breath'].append(np.random.choice([0, 1], p=[0.7, 0.3]))
            data['headache'].append(np.random.choice([0, 1], p=[0.3, 0.7])); data['chest pain'].append(np.random.choice([0, 1], p=[0.8, 0.2]))
            data['cough'].append(np.random.choice([0, 1], p=[0.3, 0.7])); data['sore throat'].append(np.random.choice([0, 1], p=[0.5, 0.5]))
            data['abdominal pain'].append(np.random.choice([0, 1], p=[0.6, 0.4])); data['nausea'].append(np.random.choice([0, 1], p=[0.5, 0.5]))
            data['fatigue'].append(np.random.choice([0, 1], p=[0.2, 0.8])); data['muscle pain'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['vakavuus'].append(1)
        else: # Vakava
            data['fever'].append(np.random.choice([0, 1], p=[0.1, 0.9])); data['shortness of breath'].append(np.random.choice([0, 1], p=[0.2, 0.8]))
            data['headache'].append(np.random.choice([0, 1], p=[0.2, 0.8])); data['chest pain'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['cough'].append(np.random.choice([0, 1], p=[0.3, 0.7])); data['sore throat'].append(np.random.choice([0, 1], p=[0.7, 0.3]))
            data['abdominal pain'].append(np.random.choice([0, 1], p=[0.5, 0.5])); data['nausea'].append(np.random.choice([0, 1], p=[0.4, 0.6]))
            data['fatigue'].append(np.random.choice([0, 1], p=[0.1, 0.9])); data['muscle pain'].append(np.random.choice([0, 1], p=[0.3, 0.7]))
            data['vakavuus'].append(2)
    df = pd.DataFrame(data)
    feature_columns = [col for col in df.columns if col != 'vakavuus']
    X = df[feature_columns]
    y = df['vakavuus']
    malli = RandomForestClassifier(n_estimators=100, random_state=42)
    malli.fit(X, y)
    return malli, feature_columns

def hae_bearer_token():
    # Tämä on täsmälleen sama funktio kuin Colabissa, mutta käyttää st.secrets.
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
            st.error(f"WHO Tokenin haku epäonnistui: {response.text}")
            return None
    except Exception as e:
        st.error(f"Odottamaton virhe WHO tokenin haussa: {e}")
        return None

def hae_who_oiretieto(hakusana_englanniksi, token):
    # Tämä on täsmälleen sama funktio kuin Colabissa.
    if not token: return {'löytyi': False, 'viesti': 'Missing Bearer token.'}
    search_url = "https://id.who.int/icd/release/11/2024-01/mms/search"
    params = {'q': hakusana_englanniksi}
    headers = {
        'Authorization': f'Bearer {token}', 'Accept': 'application/json',
        'Accept-Language': 'en', 'API-Version': 'v2'
    }
    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        if response.status_code != 200: return {'löytyi': False, 'viesti': f"Search error: {response.status_code}"}
        data = response.json()
        if not data.get('destinationEntities'): return {'löytyi': False, 'viesti': f"No results for '{hakusana_englanniksi}'."}
        entity_url = data['destinationEntities'][0].get('id', '')
        if not entity_url: return {'löytyi': False, 'viesti': 'No URL in result.'}
        detail_response = requests.get(entity_url, headers=headers, timeout=10)
        if detail_response.status_code == 200:
            detail_data = detail_response.json()
            return {
                'löytyi': True,
                'otsikko': detail_data.get('title', {}).get('@value', 'No title'),
                'koodi': data['destinationEntities'][0].get('theCode', 'No code'),
                'määritelmä': detail_data.get('definition', {}).get('@value', 'No definition.')
            }
        else: return {'löytyi': False, 'viesti': f"Error getting details: {detail_response.status_code}"}
    except Exception as e: return {'löytyi': False, 'viesti': f'Unexpected search error: {str(e)}'}

def analysoi_oireet(kayttajan_viesti, malli, piirteet, status_placeholder):
    # Tämä on pääfunktio Colabista, hieman muokattuna antamaan status-päivityksiä.
    try:
        status_placeholder.text("1/5: Haetaan WHO:n pääsylippua...")
        bearer_token = hae_bearer_token()
        if not bearer_token: return "Virhe: Ei voitu todentautua WHO:n palveluun."

        status_placeholder.text("2/5: Tunnistetaan oireita viestistä Geminin avulla...")
        gemini_tunnistus = genai.GenerativeModel('gemini-2.5-flash')
        prompt_tunnistus = f"""Analyze the following text and extract the medical symptoms. IMPORTANT: Return ONLY a comma-separated list of the symptoms in ENGLISH and in lowercase. Example: "I have a fever and a bad headache." -> "fever,headache". Text: "{kayttajan_viesti}" """
        vastaus = gemini_tunnistus.generate_content(prompt_tunnistus)
        tunnistetut_oireet_en = [o.strip().lower() for o in vastaus.text.strip().split(',') if o.strip()]
        if not tunnistetut_oireet_en: return "En valitettavasti tunnistanut viestistäsi selkeitä lääketieteellisiä oireita."

        status_placeholder.text(f"3/5: Haetaan tietoja {len(tunnistetut_oireet_en)} oireelle WHO:lta...")
        who_tiedot = [hae_who_oiretieto(oire, bearer_token) for oire in tunnistetut_oireet_en]

        status_placeholder.text("4/5: Arvioidaan vakavuutta koneoppimismallilla...")
        syote_vektori = pd.DataFrame([np.zeros(len(piirteet))], columns=piirteet)
        for oire in tunnistetut_oireet_en:
            if oire in piirteet: syote_vektori[oire] = 1
        ennuste = malli.predict(syote_vektori)[0]
        vakavuus_map = {0: "Lievä", 1: "Keskivakava", 2: "Vakava"}
        vakavuus_teksti = vakavuus_map.get(ennuste, "Tuntematon")

        status_placeholder.text("5/5: Kootaan lopullinen vastaus Geminin avulla...")
        who_teksti_en = "\n".join([f"- {t['otsikko']} (Code: {t['koodi']}): {t['määritelmä']}" for t in who_tiedot if t['löytyi']])
        prompt_lopputulos = f"""You are an empathetic AI health assistant. Synthesize the technical data below into a clear, helpful response IN FINNISH. IMPORTANT: ALWAYS remind the user that you are an AI, NOT a doctor, and they should consult a professional for serious symptoms. Technical data: - User's message: "{kayttajan_viesti}" - Symptoms identified (English): {', '.join(tunnistetut_oireet_en)} - ML model severity assessment: "{vakavuus_teksti}" - WHO information (English): {who_teksti_en}"""
        gemini_koonti = genai.GenerativeModel('gemini-2.5-flash')
        lopullinen_vastaus = gemini_koonti.generate_content(prompt_lopputulos)
        status_placeholder.empty()
        return lopullinen_vastaus.text
    except Exception as e:
        st.error(f"Kriittinen virhe analyysiprosessissa: {e}")
        return "Pahoittelut, teknisen virheen vuoksi en pystynyt käsittelemään pyyntöäsi."

# --- KÄYTTÖLIITTYMÄN TOIMINNALLISUUS ---

# Ladataan malli kerran, kun sovellus käynnistyy
vakavuusmalli, piirteet = luo_ja_kouluta_malli()

user_input = st.text_area("Syötä oireesi tähän:", "I feel very tired, I have a strong headache and a bit of a cough.", height=100)

if st.button("Analysoi oireet"):
    if user_input:
        # Luodaan tyhjä elementti, jota voidaan päivittää status-viesteillä
        status_placeholder = st.empty()
        with st.spinner('Analysoidaan...'):
            tulos = analysoi_oireet(user_input, vakavuusmalli, piirteet, status_placeholder)
            st.markdown("---")
            st.markdown("### Analyysin tulokset:")
            st.markdown(tulos)
    else:

        st.warning("Syötä oireesi ennen analysointia.")
