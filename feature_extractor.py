"""
feature_extractor.py
Extracts features matching PhiUSIIL dataset column distributions exactly.
Key insight from dataset analysis:
  - Safe URLs:    URLSimilarityIndex=100, IsHTTPS=1, CharContinuationRate≈1.0
  - Phishing URLs: URLSimilarityIndex≈50, IsHTTPS≈0.5, CharContinuationRate≈0.73
"""

import re, math, urllib.parse
from collections import Counter

SUSPICIOUS_KEYWORDS = [
    'login','signin','verify','update','secure','account','banking',
    'confirm','password','credential','wallet','paypal','ebay','amazon',
    'apple','microsoft','google','facebook','netflix','support','auth',
    'validate','urgent','alert','suspended','limited','expire','recover'
]
SHORTENERS = ['bit.ly','tinyurl.com','goo.gl','t.co','ow.ly','is.gd','buff.ly','tiny.cc','cutt.ly']
SUSPICIOUS_TLDS = {'.tk','.ml','.ga','.cf','.gq','.xyz','.top','.club','.online','.site','.work','.loan','.party','.racing'}
COMMON_TLDS = {'com','org','net','edu','gov','io','co','uk','de','fr','jp','au','ca','us','info','me','tv','biz'}

# Official domains of common brands — URL similarity = 100 only for these
LEGIT_BRAND_DOMAINS = {
    'paypal.com','amazon.com','google.com','facebook.com','apple.com',
    'microsoft.com','netflix.com','ebay.com','instagram.com','twitter.com',
    'linkedin.com','chase.com','bankofamerica.com','wellsfargo.com',
    'github.com','stackoverflow.com','youtube.com','reddit.com',
    'wikipedia.org','yahoo.com','bing.com','dropbox.com','spotify.com',
}

def _entropy(s):
    if not s: return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())

def _get_registered_domain(domain):
    """Get the registrable domain (e.g. 'www.google.com' -> 'google.com')"""
    parts = domain.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return domain

def extract_features(url: str) -> dict:
    url = url.strip()
    if not url.startswith(('http://','https://')):
        url = 'http://' + url

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower().split(':')[0]
        path   = parsed.path
        query  = parsed.query
    except Exception:
        domain, path, query = '', '', ''

    full  = url.lower()
    parts = domain.split('.')
    tld   = parts[-1] if parts else ''
    reg_domain = _get_registered_domain(domain)

    # ── URLLength ────────────────────────────────────────────────────────
    url_length = len(url)

    # ── DomainLength ────────────────────────────────────────────────────
    domain_length = len(domain)

    # ── IsDomainIP ──────────────────────────────────────────────────────
    is_domain_ip = 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain) else 0

    # ── TLDLegitimateProb ───────────────────────────────────────────────
    tld_legit_prob = 0.85 if tld in COMMON_TLDS else 0.15

    # ── URLCharProb — ratio of alphanumeric chars ────────────────────────
    # Safe URLs: higher ratio (~0.8).  Phishing: lower due to hyphens/special
    alnum = sum(c.isalnum() for c in url)
    url_char_prob = round(alnum / max(len(url), 1), 6)

    # ── TLDLength ───────────────────────────────────────────────────────
    tld_length = len(tld)

    # ── NoOfSubDomain ───────────────────────────────────────────────────
    no_of_subdomain = max(0, len(parts) - 2)

    # ── HasObfuscation / NoOfObfuscatedChar / ObfuscationRatio ──────────
    obf_chars = len(re.findall(r'%[0-9a-fA-F]{2}', url))
    has_obf   = 1 if obf_chars > 0 else 0
    obf_ratio = round(obf_chars / max(len(url), 1), 4)

    # ── Letters ─────────────────────────────────────────────────────────
    letters      = sum(c.isalpha() for c in url)
    letter_ratio = round(letters / max(len(url), 1), 4)

    # ── Digits ──────────────────────────────────────────────────────────
    digits      = sum(c.isdigit() for c in url)
    digit_ratio = round(digits / max(len(url), 1), 4)

    # ── Query params ────────────────────────────────────────────────────
    no_equals    = url.count('=')
    no_qmark     = url.count('?')
    no_ampersand = url.count('&')

    # ── Special chars ────────────────────────────────────────────────────
    no_other_special = sum(1 for c in url if not c.isalnum() and c not in ':/.-_?=&%#+@')
    total_special    = sum(1 for c in url if not c.isalnum())
    spacial_ratio    = round(total_special / max(len(url), 1), 4)

    # ── IsHTTPS ─────────────────────────────────────────────────────────
    is_https = 1 if parsed.scheme == 'https' else 0

    # ── NoOfURLRedirect ─────────────────────────────────────────────────
    no_redirect = 1 if '//' in path else 0

    # ── URLSimilarityIndex ───────────────────────────────────────────────
    # Dataset: safe=100 always, phishing≈50 (0-100 Levenshtein-based)
    # Heuristic:
    #   - If registered domain matches a known legit domain exactly → 100
    #   - If a brand name appears in the domain but it's NOT the legit domain → low (~30-50)
    #   - No brand match at all → 100 (it's just an unknown site, not spoofing)
    brands = ['paypal','amazon','google','facebook','apple','microsoft',
              'netflix','ebay','instagram','twitter','linkedin','chase',
              'wellsfargo','bankofamerica','citibank','dropbox','spotify']
    brand_in_domain = [b for b in brands if b in domain]
    if reg_domain in LEGIT_BRAND_DOMAINS:
        url_similarity = 100.0   # genuine brand domain
    elif brand_in_domain and reg_domain not in LEGIT_BRAND_DOMAINS:
        url_similarity = 30.0    # spoofed brand — very suspicious
    else:
        url_similarity = 100.0   # unknown site, not a spoof

    # ── CharContinuationRate ─────────────────────────────────────────────
    # Dataset: safe≈0.93-1.0, phishing≈0.73
    # Heuristic: count character transitions in domain
    if len(domain) < 2:
        char_cont = 0.0
    else:
        # Count positions where char class stays same (letter→letter or digit→digit)
        same = sum(1 for i in range(1, len(domain))
                   if (domain[i].isalpha() == domain[i-1].isalpha()) and domain[i] != '.')
        char_cont = round(same / max(len(domain)-1, 1), 4)

    # ── Extra UI-only fields (prefixed _ so app knows not to feed to model) ─
    features = {
        # ── MODEL FEATURES (match PhiUSIIL column names exactly) ──────
        'URLLength':                  url_length,
        'DomainLength':               domain_length,
        'IsDomainIP':                 is_domain_ip,
        'TLDLegitimateProb':          tld_legit_prob,
        'URLCharProb':                url_char_prob,
        'TLDLength':                  tld_length,
        'NoOfSubDomain':              no_of_subdomain,
        'HasObfuscation':             has_obf,
        'NoOfObfuscatedChar':         obf_chars,
        'ObfuscationRatio':           obf_ratio,
        'NoOfLettersInURL':           letters,
        'LetterRatioInURL':           letter_ratio,
        'NoOfDegitsInURL':            digits,
        'DegitRatioInURL':            digit_ratio,
        'NoOfEqualsInURL':            no_equals,
        'NoOfQMarkInURL':             no_qmark,
        'NoOfAmpersandInURL':         no_ampersand,
        'NoOfOtherSpecialCharsInURL': no_other_special,
        'SpacialCharRatioInURL':      spacial_ratio,
        'IsHTTPS':                    is_https,
        'NoOfURLRedirect':            no_redirect,
        'Bank':                       1 if any(k in full for k in ['bank','banking']) else 0,
        'Pay':                        1 if any(k in full for k in ['pay','payment','paypal','checkout']) else 0,
        'HasPasswordField':           1 if 'password' in full or 'passwd' in full else 0,
        'HasHiddenFields':            1 if 'hidden' in full else 0,
        'HasExternalFormSubmit':      1 if any(k in full for k in ['submit','login','signin']) else 0,
        'URLSimilarityIndex':         url_similarity,
        'CharContinuationRate':       char_cont,

        # ── UI-ONLY fields (not fed to model) ─────────────────────────
        '_url_entropy':    round(_entropy(url), 4),
        '_domain_entropy': round(_entropy(domain), 4),
        '_num_hyphens':    url.count('-'),
        '_num_dots':       url.count('.'),
        '_has_shortener':  1 if any(s in domain for s in SHORTENERS) else 0,
        '_has_at':         1 if '@' in url else 0,
        '_susp_tld':       1 if ('.'+tld) in SUSPICIOUS_TLDS else 0,
        '_keyword_count':  sum(k in full for k in SUSPICIOUS_KEYWORDS),
        '_reg_domain':     reg_domain,
    }
    return features


def features_to_vector(features: dict, feature_cols: list) -> list:
    return [float(features.get(c, 0)) for c in feature_cols]


def get_suspicious_indicators(features: dict, url: str) -> list:
    inds = []
    if features.get('IsDomainIP'):
        inds.append(('danger', 'Uses raw IP address instead of a domain name'))
    if not features.get('IsHTTPS'):
        inds.append(('warning', 'No HTTPS — connection is unencrypted'))
    if features.get('_susp_tld'):
        inds.append(('danger', 'Suspicious free/high-risk TLD detected'))
    if features.get('URLSimilarityIndex', 100) <= 30:
        inds.append(('danger', 'Brand name spoofed in domain (e.g. paypal-secure.tk)'))
    if features.get('NoOfSubDomain', 0) >= 3:
        inds.append(('warning', f"Excessive subdomains ({features['NoOfSubDomain']})"))
    if features.get('_keyword_count', 0) > 0:
        inds.append(('warning', f"Contains {features['_keyword_count']} phishing keyword(s)"))
    if features.get('_has_at'):
        inds.append(('danger', 'Contains @ symbol — browser redirect trick'))
    if features.get('NoOfURLRedirect'):
        inds.append(('danger', 'Double-slash redirect detected in path'))
    if features.get('HasObfuscation'):
        inds.append(('warning', f"URL obfuscation/encoding detected"))
    if features.get('_num_hyphens', 0) >= 4:
        inds.append(('warning', f"Many hyphens ({features['_num_hyphens']}) — common in phishing domains"))
    if features.get('URLLength', 0) > 100:
        inds.append(('warning', f"Very long URL ({features['URLLength']} chars)"))
    if features.get('HasExternalFormSubmit'):
        inds.append(('info', 'Login/submit keywords in URL'))
    if features.get('Bank') or features.get('Pay'):
        inds.append(('info', 'Financial keywords (bank/pay) in URL'))
    return inds


def calculate_risk_score(features: dict, label: str, confidence: float) -> int:
    score = 0
    score += confidence * 55 if label == 'Phishing' else (1 - confidence) * 20
    if features.get('IsDomainIP'):                     score += 15
    if not features.get('IsHTTPS'):                    score += 10
    if features.get('_susp_tld'):                      score += 12
    if features.get('URLSimilarityIndex', 100) <= 30:  score += 18
    if features.get('_has_at'):                        score += 8
    score += min(features.get('_keyword_count', 0) * 3, 12)
    if features.get('NoOfSubDomain', 0) >= 3:          score += 6
    if features.get('HasObfuscation'):                 score += 5
    return min(100, round(score))
