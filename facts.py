"""Deterministic facts layer — the part that must NOT be fine-tuned.

Handles the knowledge an LLM should never be trusted to memorise:
  * today's date          -> injected into prompts at request time
  * Indian institutions   -> fuzzy-corrected from speech-recognition errors
  * PIN codes             -> validated, state auto-filled

See finetune/GUIDE.md, Lesson 1, for why this is a lookup table and not training data.
"""
import json
import os
import re
import sqlite3
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import httpx

REF_DIR = Path(__file__).parent / "data" / "reference"

# A candidate name must be at least this similar to a canonical one to be corrected.
# Lower = more corrections but more wrong ones. 0.62 catches "Thapadi"->"Thapar" while
# leaving genuinely unknown colleges untouched.
# A type mismatch costs a match a fifth of its score (see correct_institution), so a
# genuine correction across the university/institute line lands just under the old
# 0.62: "Thapadi University" scores 0.77 against Thapar Institute of Engineering and
# Technology, and 0.616 once penalised.
MATCH_THRESHOLD = 0.60

_institutions: list[str] | None = None
_pin_ranges: dict[str, str] | None = None

LANG_NAMES = {"en": "English", "hi": "Hindi", "pa": "Punjabi"}


def institutions() -> list[str]:
    """Canonical institution names, or an empty list if the reference file is absent.

    A missing or malformed reference file must never abort a finished interview: the
    correction is an improvement, not a prerequisite. Without it names are simply left
    exactly as the candidate said them.
    """
    global _institutions
    if _institutions is None:
        try:
            data = json.loads((REF_DIR / "universities.json").read_text())
            _institutions = list(data["institutions"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            print("facts: universities.json unavailable - skipping institution correction")
            _institutions = []
    return _institutions # type:ignore


def pin_ranges() -> dict[str, str]:
    global _pin_ranges
    if _pin_ranges is None:
        try:
            data = json.loads((REF_DIR / "pincode_ranges.json").read_text())
            _pin_ranges = {k: v for k, v in data.items() if not k.startswith("_")}
        except (OSError, json.JSONDecodeError, AttributeError):
            # Same contract as institutions(): a missing reference file costs accuracy,
            # never an interview. Without it the offline fallback simply declines to
            # guess a state.
            print("facts: pincode_ranges.json unavailable - no offline PIN fallback")
            _pin_ranges = {}
    return _pin_ranges


# ---------------------------------------------------------------- date awareness

def date_context() -> str:
    """Text appended to system prompts so the model can resolve relative dates.

    Never train a date into weights - it is wrong the day after training.
    """
    today = date.today()
    academic_year = f"{today.year}-{str(today.year + 1)[2:]}" if today.month >= 6 else \
                    f"{today.year - 1}-{str(today.year)[2:]}"
    return (
        f"\n\nToday's date is {today:%d %B %Y}. The current academic year in India is "
        f"{academic_year}. Use this to resolve relative dates the candidate mentions "
        f"(for example 'two years back', 'last semester', 'final year') into actual years."
    )


# ------------------------------------------------------- institution correction

# Words that carry no distinguishing information in an institution name. Speech
# recognition frequently misspells the long ones ("univarsity", "collage"), so these
# are matched fuzzily below rather than exactly.
_GENERIC_WORDS = ("university", "universities", "vishwavidyalaya", "vidyapeeth",
                  "college", "institute", "technology", "engineering", "school")
_SHORT_STOPWORDS = {"of", "the", "and", "for", "in", "at"}


# Kinds of institution that are not each other, whatever the fuzzy score says. The
# correction was quietly changing the kind: "ITI Hamirpur" became "National Institute of
# Technology Hamirpur", turning a man with a trade certificate into an NIT graduate on a
# document he hands to an employer. It matched on the city alone, because the words that
# told the two apart had been stripped as generic before the comparison.
#
# Only the hard boundaries are listed. University, college, institute and vidyapeeth are
# deliberately absent: Indian usage moves freely between them, and Thapar Institute of
# Engineering and Technology is universally called Thapar University by the people who
# went there.
_EXCLUSIVE_TYPES = ("iti", "polytechnic", "school")


def _institution_type(name: str) -> str | None:
    """A kind of institution that cannot be corrected into any other kind."""
    for token in re.findall(r"[a-z]+", name.lower()):
        for marker in _EXCLUSIVE_TYPES:
            if token == marker:
                return marker
            # "polytecnic", "politechnic" - long enough to misspell, so matched fuzzily.
            if len(token) >= 6 and SequenceMatcher(None, token, marker).ratio() >= 0.85:
                return marker
    return None


def _generic_word(name: str) -> str | None:
    """The "university"/"college"/"institute" in a name, however it was spelt."""
    for token in re.findall(r"[a-z]+", name.lower()):
        if len(token) < 6:
            continue
        for word in _GENERIC_WORDS:
            if SequenceMatcher(None, token, word).ratio() >= 0.80:
                return "university" if word in ("universities", "vishwavidyalaya",
                                                "vidyapeeth") else word
    return None


def _strip_generic(token: str) -> bool:
    """Is this token a generic institution word, even if it is misspelt?"""
    if token in _SHORT_STOPWORDS:
        return True
    if len(token) < 6:
        return False
    return any(SequenceMatcher(None, token, word).ratio() >= 0.80
               for word in _GENERIC_WORDS)


# Colloquial campus city names. Applied to both sides of the comparison, so a
# candidate saying "NIT Trichy" lines up with the official "... Tiruchirappalli".
_CITY_ALIASES = (
    (r"\btrichy\b", "tiruchirappalli"),
    (r"\bbangalore\b", "bengaluru"),
    (r"\bbombay\b", "bombay"),
    (r"\bcalcutta\b", "calcutta"),
    (r"\bvizag\b", "visakhapatnam"),
    (r"\bbbsr\b", "bhubaneswar"),
    (r"\bkgp\b", "kharagpur"),
    (r"\bvaranasi\b", "varanasi bhu"),
)


def _norm(s: str) -> str:
    s = s.lower()
    # "i i t rurkee" -> "iit rurkee". Whisper reliably spells initialisms out as
    # separate letters, which would otherwise never match the abbreviations below.
    s = re.sub(r"\b(?:[a-z]\s+){1,6}[a-z]\b", lambda m: m.group(0).replace(" ", ""), s)
    for old, new in _CITY_ALIASES:
        s = re.sub(old, new, s)
    # expand common spoken/abbreviated forms so fuzzy matching lines up
    for short, full in (
    # Special IIIT names
    (r"\biiit[\s-]*(?:d|delhi)\b",
     "indraprastha institute of information technology delhi"),
    (r"\biiit[\s-]*(?:h|hyd|hyderabad)\b",
     "international institute of information technology hyderabad"),
    (r"\biiit[\s-]*(?:b|blr|bangalore|bengaluru)\b",
     "international institute of information technology bangalore"),

    # Institution families
    (r"\biiit\b", "indian institute of information technology"),
    (r"\biit\b", "indian institute of technology"),
    (r"\bnit\b", "national institute of technology"),
    (r"\biim\b", "indian institute of management"),
    (r"\biisc\b", "indian institute of science"),
    (r"\biiser\b", "indian institute of science education and research"),
    (r"\baiims\b", "all india institute of medical sciences"),
    (r"\bnlu\b", "national law university"),
    (r"\bniper\b", "national institute of pharmaceutical education and research"),
    (r"\bnift\b", "national institute of fashion technology"),
    (r"\bnid\b", "national institute of design"),
    (r"\bspa\b", "school of planning and architecture"),
    (r"\bbits\b", "birla institute of technology and science"),

    # National and central institutions
    (r"\biist\b", "indian institute of space science and technology"),
    (r"\biiest\b", "indian institute of engineering science and technology"),
    (r"\bniser\b", "national institute of science education and research"),
    (r"\bisi\b", "indian statistical institute"),
    (r"\btiss\b", "tata institute of social sciences"),
    (r"\btifr\b", "tata institute of fundamental research"),
    (r"\bjnu\b", "jawaharlal nehru university"),
    (r"\bbhu\b", "banaras hindu university"),
    (r"\bamu\b", "aligarh muslim university"),
    (r"\bjmi\b", "jamia millia islamia"),
    (r"\bignou\b", "indira gandhi national open university"),
    (r"\b(?:uoh|hcu)\b", "university of hyderabad"),
    (r"\beflu\b", "english and foreign languages university"),
    (r"\bmanuu\b", "maulana azad national urdu university"),
    (r"\bnehu\b", "north eastern hill university"),
    (r"\bhnbgu\b", "hemvati nandan bahuguna garhwal university"),
    (r"\bign?tu\b", "indira gandhi national tribal university"),

    # Delhi and northern India
    (r"\bdtu\b", "delhi technological university"),
    (r"\bnsut\b", "netaji subhas university of technology"),
    (r"\bigdtuw\b", "indira gandhi delhi technical university for women"),
    (r"\b(?:ggsipu|ipu)\b",
     "guru gobind singh indraprastha vishwavidyalaya"),
    (r"\bgndu\b", "guru nanak dev university"),
    (r"\blpu\b", "lovely professional university"),

    # Technical universities
    (r"\b(?:aktu|uptu)\b",
     "dr apj abdul kalam technical university"),
    (r"\bktu\b", "apj abdul kalam technological university"),
    (r"\bvtu\b", "visvesvaraya technological university"),
    (r"\b(?:makaut|wbut)\b",
     "maulana abul kalam azad university of technology"),
    (r"\bjntuh\b", "jawaharlal nehru technological university hyderabad"),
    (r"\bjntuk\b", "jawaharlal nehru technological university kakinada"),
    (r"\bjntua\b", "jawaharlal nehru technological university anantapur"),
    (r"\bgtu\b", "gujarat technological university"),
    (r"\brtu\b", "rajasthan technical university"),
    (r"\brgpv\b", "rajiv gandhi proudyogiki vishwavidyalaya"),
    (r"\bbput\b", "biju patnaik university of technology"),
    (r"\bcsvtu\b", "chhattisgarh swami vivekanand technical university"),
    (r"\bhbtu\b", "harcourt butler technical university"),
    (r"\bikgptu\b", "ik gujral punjab technical university"),
    (r"\bsppu\b", "savitribai phule pune university"),
    (r"\bcusat\b", "cochin university of science and technology"),

    # Law universities
    (r"\bnlsiu\b", "national law school of india university"),
    (r"\bnalsar\b", "nalsar university of law"),
    (r"\bnlud\b", "national law university delhi"),
    (r"\bnliu\b", "national law institute university"),
    (r"\bgnlu\b", "gujarat national law university"),
    (r"\bhnlu\b", "hidayatullah national law university"),
    (r"\brgnul\b", "rajiv gandhi national university of law"),
    (r"\bcnlu\b", "chanakya national law university"),
    (r"\bnuals\b", "national university of advanced legal studies"),
    (r"\bnluo\b", "national law university odisha"),
    (r"\bnusrl\b", "national university of study and research in law"),

    # Private and deemed universities
    (r"\bvit\b", "vellore institute of technology"),
    (r"\b(?:srm|srmist)\b",
     "srm institute of science and technology"),
    (r"\bmahe\b", "manipal academy of higher education"),
    (r"\btiet\b", "thapar institute of engineering and technology"),
    (r"\bkiit\b", "kalinga institute of industrial technology"),
    (r"\bsoa\b", "siksha o anusandhan"),
    (r"\bgitam\b", "gandhi institute of technology and management"),
    (r"\bsastra\b",
     "shanmugha arts science technology and research academy"),
    (r"\bnmims\b", "narsee monjee institute of management studies"),
    (r"\bupes\b", "university of petroleum and energy studies"),

    (r"\bpu\s+(?:chandigarh|panjab)\b", "panjab university"),
    (r"\bpu\s+patna\b", "patna university"),
    (r"\bpu\s+(?:pondicherry|puducherry)\b", "pondicherry university"),
    (r"\bcu\s+chandigarh\b", "chandigarh university"),
    (r"\bcu\s+(?:calcutta|kolkata)\b", "university of calcutta"),
    (r"\bdu\s+(?:delhi|new delhi)\b", "university of delhi"),
    (r"\bju\s+(?:jadavpur|kolkata)\b", "jadavpur university"),
    (r"\bou\s+hyderabad\b", "osmania university"),
    (r"\bau\s+chennai\b", "anna university"),
    ):
        s = re.sub(short, full, s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    # Drop generic words last, fuzzily, so misspellings do not survive as fake evidence.
    return " ".join(t for t in s.split() if not _strip_generic(t)).strip()


def correct_institution(name: str) -> tuple[str, float]:
    """Map a possibly-garbled institution name to a canonical one.

    Returns (name, score). If nothing is similar enough the input is returned
    unchanged with its best score - we never invent an institution.
    """
    if not name or not name.strip():
        return name, 0.0
    target = _norm(name)
    if not target:
        return name, 0.0
    target_tokens = set(target.split())
    spoken_type = _institution_type(name)
    raw = name.lower()
    spoken_generic = _generic_word(name)
    # (score, closeness) - see the tie-break below.
    best, best_score, best_raw = name, 0.0, 0.0
    for canonical in institutions():
        # Never correct across a hard type boundary: an ITI is not an NIT.
        if _institution_type(canonical) != spoken_type:
            continue
        canonical_norm = _norm(canonical)
        score = SequenceMatcher(None, target, canonical_norm).ratio()
        # Shared tokens are strong evidence, but only in proportion to how much of the
        # name they actually cover. A flat boost for any single shared word makes every
        # "Guru ..." or "National ..." institution score identically, and the winner is
        # then decided by list order rather than by similarity.
        canonical_tokens = set(canonical_norm.split())
        overlap = target_tokens & canonical_tokens
        if overlap:
            coverage = len(overlap) / max(len(target_tokens), len(canonical_tokens))
            score = max(score, coverage)
        # Stripping the generic words is what lets "Thapar Institute" reach its official
        # name, but it also makes "Guru Nanak Dev University" and "Guru Nanak Dev
        # Engineering College" identical, and "Punjab Univercity" score a perfect 1.0
        # against "Punjab Engineering College". When two canonical names score the same,
        # the one that actually looks like what the candidate said wins.
        closeness = SequenceMatcher(None, raw, canonical.lower()).ratio()
        # Stripping "university" and "college" is what lets "Thapar Institute" reach its
        # official name, but it also throws away the only thing telling Panjab University
        # from Punjab Engineering College - so "Punjab Univercity" matched the college at
        # a perfect 1.0. Put that word back as a penalty rather than a filter: naming a
        # different kind counts against a match without forbidding it, because people do
        # call Thapar Institute a university and are not wrong about which place they mean.
        if spoken_generic and (other := _generic_word(canonical)) and other != spoken_generic:
            score *= 0.8
        if (score, closeness) > (best_score, best_raw):
            best, best_score, best_raw = canonical, score, closeness
    return (best, best_score) if best_score >= MATCH_THRESHOLD else (name, best_score)


# ------------------------------------------------------------- pincode handling

PIN_RE = re.compile(r"\b(\d{6})\b")
# "PIN 140401", "pin code 140401", "pincode- 140401" as spoken and transcribed.
_LABELLED_PIN = re.compile(r"\b(?:pin|pincode|pin\s*code|postal\s*code)\b[\s:.-]*",
                           re.IGNORECASE)

# Department of Posts' All India Pincode Directory, exposed through data.gov.in.
# The API key below is the public key published with the resource; deployments can
# override it with their own data.gov.in key without changing the code.
PINCODE_API_URL = (
    "https://api.data.gov.in/resource/5c2f62fe-5afa-4119-a499-fec9d604d5bd"
)
PINCODE_SECONDARY_API_URL = "https://api.postalpincode.in/pincode/{pin}"
PINCODE_DB_PATH = Path(
    os.getenv("PINCODE_DB_PATH", str(REF_DIR / "india_post_pincodes.sqlite3"))
)
PINCODE_API_KEY = os.getenv(
    "DATA_GOV_IN_API_KEY",
    "579b464db66ec23bdd000001cdc3b564546246a772a26393094f5645",
)
PINCODE_API_TIMEOUT = 5.0
PINCODE_MATCH_THRESHOLD = 0.72


class PincodeServiceUnavailable(RuntimeError):
    """Raised when the postal directory cannot provide a trustworthy response."""


class PincodeDatabaseUnavailable(RuntimeError):
    """Raised when the optional local postal snapshot cannot be queried."""


_PLACE_ALIASES = (
    (r"\bbangalore\b", "bengaluru"),
    (r"\bbombay\b", "mumbai"),
    (r"\bcalcutta\b", "kolkata"),
    (r"\bcochin\b", "kochi"),
    (r"\bgauhati\b", "guwahati"),
    (r"\bgurgaon\b", "gurugram"),
    (r"\bmadras\b", "chennai"),
    (r"\bmysore\b", "mysuru"),
    (r"\borissa\b", "odisha"),
    (r"\bpondicherry\b", "puducherry"),
    (r"\bpoona\b", "pune"),
    (r"\btrivandrum\b", "thiruvananthapuram"),
    (r"\bs(?:ahibzada)?\s*a(?:jit)?\s*s(?:ingh)?\s+nagar\b", "mohali"),
)


def _normalise_place(value: str, pin: str = "") -> str:
    """Normalise address/place text for a conservative local comparison."""
    value = (value or "").lower()
    if pin:
        value = re.sub(rf"\b{re.escape(pin)}\b", " ", value)
    for old, new in _PLACE_ALIASES:
        value = re.sub(old, new, value)
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    # These words say nothing about the actual place. Keep terms such as sector,
    # road and village because they can occur in a post-office name.
    value = re.sub(
        r"\b(?:address|flat|floor|house|india|no|number|pin|pincode|postal|code)\b",
        " ",
        value,
    )
    return re.sub(r"\s+", " ", value).strip()


def _place_score(text: str, candidate: str) -> float:
    """Score a postal place name against the user-supplied address text."""
    if not text or not candidate:
        return 0.0
    if candidate in text or text in candidate:
        return 1.0
    text_tokens = set(text.split())
    candidate_tokens = set(candidate.split())
    overlap = text_tokens & candidate_tokens
    coverage = len(overlap) / len(candidate_tokens) if candidate_tokens else 0.0
    return max(coverage, SequenceMatcher(None, text, candidate).ratio())


# India Post writes a district as "Hamirpur(Hp)" - the name with the state bolted on. Left
# as it was, the comparison against what the candidate said never matched and the resume
# read "Hamirpur, 177001, Himachal Pradesh, Hamirpur(Hp)".
_POSTAL_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def _strip_postal_suffix(value: str) -> str:
    return _POSTAL_SUFFIX.sub("", value or "").strip()


def _display_place(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().title()


@lru_cache(maxsize=4096)
def _fetch_local_post_offices(pin: str) -> tuple[dict, ...]:
    """Read one PIN from the local indexed Department of Posts snapshot."""
    if not PINCODE_DB_PATH.is_file():
        raise PincodeDatabaseUnavailable("Local postal database is not installed")

    try:
        database_uri = f"file:{PINCODE_DB_PATH.resolve()}?mode=ro&immutable=1"
        with sqlite3.connect(database_uri, uri=True, timeout=1.0) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT circlename, regionname, divisionname, officename, pincode,
                       officetype, delivery, district, statename
                FROM post_offices
                WHERE pincode = ?
                """,
                (pin,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise PincodeDatabaseUnavailable("Local postal database is unreadable") from exc
    return tuple(dict(row) for row in rows)


@lru_cache(maxsize=2048)
def _fetch_post_offices(pin: str) -> tuple[dict, ...]:
    """Fetch Department of Posts records for one PIN; cache to reduce latency.

    Only the PIN is sent to data.gov.in. The candidate's full address remains local.
    An empty tuple is an authoritative "not found" response. Service/schema errors
    raise PincodeServiceUnavailable so callers can distinguish them from invalid PINs.
    """
    try:
        response = httpx.get(
            PINCODE_API_URL,
            params={
                "api-key": PINCODE_API_KEY,
                "format": "json",
                "limit": 100,
                "filters[pincode]": pin,
            },
            headers={"Accept": "application/json", "User-Agent": "TePMA/1.0"},
            timeout=PINCODE_API_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PincodeServiceUnavailable("Postal PIN service is unavailable") from exc

    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise PincodeServiceUnavailable("Unexpected postal PIN service response")

    records = payload.get("records")
    if not isinstance(records, list):
        raise PincodeServiceUnavailable("Postal PIN records are missing")

    # Ignore malformed or mismatched records rather than trusting the response blindly.
    return tuple(
        record for record in records
        if isinstance(record, dict) and str(record.get("pincode", "")) == pin
    )


@lru_cache(maxsize=2048)
def _fetch_secondary_post_offices(pin: str) -> tuple[dict, ...]:
    """Use the public postal directory only when data.gov.in is unavailable."""
    try:
        response = httpx.get(
            PINCODE_SECONDARY_API_URL.format(pin=pin),
            headers={"Accept": "application/json", "User-Agent": "TePMA/1.0"},
            timeout=PINCODE_API_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise PincodeServiceUnavailable("Secondary postal service is unavailable") from exc

    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise PincodeServiceUnavailable("Unexpected secondary postal response")
    result = payload[0]
    if str(result.get("Status", "")).lower() != "success":
        return ()
    offices = result.get("PostOffice")
    if not isinstance(offices, list):
        raise PincodeServiceUnavailable("Secondary postal records are missing")

    records = []
    for office in offices:
        if not isinstance(office, dict) or str(office.get("Pincode", "")) != pin:
            continue
        # Convert the secondary schema once so all matching logic below remains
        # source-independent and easy to test.
        records.append({
            "circlename": office.get("Circle"),
            "regionname": office.get("Region"),
            "divisionname": office.get("Division"),
            "officename": office.get("Name"),
            "pincode": str(office.get("Pincode", "")),
            "officetype": office.get("BranchType"),
            "delivery": office.get("DeliveryStatus"),
            "district": office.get("District"),
            "statename": office.get("State"),
        })
    return tuple(records)


def _fallback_pincode(pin: str) -> dict | None:
    """Return the old prefix-based result when live verification is unavailable."""
    try:
        region = pin_ranges().get(pin[:2])
    except (OSError, json.JSONDecodeError):
        return None
    if not region:
        return None
    return {
        "pincode": pin,
        "state": region,
        "district": None,
        "districts": [],
        "post_offices": [],
        "place_match": None,
        "place_match_score": 0.0,
        "match_field": None,
        "matched_place": None,
        "verified": False,
        "confidence": 0.25,
        "source": "local_prefix_fallback",
    }


def lookup_pincode(text: str) -> dict | None:
    """Validate a PIN and compare its postal places with the supplied address.

    Returns None when there is no PIN or the postal API confirms it does not exist.
    A result with ``verified=False`` means the API was unavailable and only the old,
    coarse two-digit prefix fallback could be used.
    """
    m = PIN_RE.search(text or "")
    if not m:
        return None
    pin = m.group(1)
    if pin[0] == "0":  # Indian PINs never start with 0
        return None

    source = "local_department_of_posts_snapshot"
    confidence = 1.0
    try:
        records = _fetch_local_post_offices(pin)
    except PincodeDatabaseUnavailable:
        records = ()

    # A local hit never needs the network. A miss checks the API because a newly
    # allocated PIN may not exist in an older snapshot.
    if not records:
        source = "department_of_posts_data_gov_in"
        try:
            records = _fetch_post_offices(pin)
        except PincodeServiceUnavailable:
            try:
                records = _fetch_secondary_post_offices(pin)
                source = "postalpincode_in_fallback"
                confidence = 0.95
            except PincodeServiceUnavailable:
                return _fallback_pincode(pin)
    if not records:  # The API responded successfully and found no such PIN.
        return None

    address = _normalise_place(text, pin)
    field_weights = {
        "officename": 1.0,
        "district": 0.95,
        "divisionname": 0.85,
        "statename": 0.80,
        "circlename": 0.75,
    }
    best_score = 0.0
    best_record: dict | None = None
    best_field: str | None = None
    best_place: str | None = None
    best_candidate_length = 0

    for record in records:
        for field, weight in field_weights.items():
            raw_place = str(record.get(field) or "")
            candidate = _normalise_place(raw_place)
            # Directory suffixes do not form part of a place as people say it.
            candidate = re.sub(r"\b(?:circle|division|gpo|ho|so|bo|po)$", "", candidate).strip()
            score = _place_score(address, candidate) * weight
            # Prefer a more specific place when two exact fields both match, e.g.
            # "Connaught Place" over the more general "New Delhi".
            if score > best_score or (
                score == best_score and len(candidate) > best_candidate_length
            ):
                best_score = score
                best_record = record
                best_field = field
                best_place = raw_place
                best_candidate_length = len(candidate)

    states = sorted({
        _display_place(str(record.get("statename") or ""))
        for record in records if record.get("statename")
    })
    districts = sorted({
        _display_place(str(record.get("district") or ""))
        for record in records if record.get("district")
    })
    post_offices = sorted({
        _display_place(str(record.get("officename") or ""))
        for record in records if record.get("officename")
    })

    # Prefer the district attached to the best local match. For a state-only match,
    # selecting one district from a multi-district PIN would imply false precision.
    district = None
    if best_record is not None and best_field in {"officename", "district", "divisionname"}:
        district = _display_place(str(best_record.get("district") or "")) or None
    elif len(districts) == 1:
        district = districts[0]

    has_place_text = bool(address)
    place_match = None if not has_place_text else best_score >= PINCODE_MATCH_THRESHOLD
    return {
        "pincode": pin,
        "state": states[0] if states else None,
        "district": district,
        "districts": districts,
        "post_offices": post_offices,
        "place_match": place_match,
        "place_match_score": round(best_score, 3),
        "match_field": best_field if place_match else None,
        "matched_place": _display_place(best_place or "") if place_match else None,
        "verified": True,
        "confidence": confidence,
        "source": source,
    }


# ------------------------------------------------------------ phone normalisation

def normalise_phone(raw: str) -> str:
    """Format an Indian mobile number as '+91 XXXXXXXXXX'.

    Deterministic formatting belongs here, not in the prompt: an LLM asked to format
    numbers will occasionally drop or invent a digit, whereas this cannot.
    Anything that is not a recognisable 10-digit Indian number is returned unchanged.
    """
    if not raw:
        return raw
    digits = re.sub(r"\D", "", raw)
    for prefix in ("91", "091", "0"):  # country code or trunk prefix
        if len(digits) > 10 and digits.startswith(prefix):
            digits = digits[len(prefix):]
            break
    if len(digits) == 10 and digits[0] in "6789":  # Indian mobiles start 6-9
        return f"+91 {digits}"
    return raw


# ---------------------------------------------------------- email normalisation

EMAIL_RE = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~.-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+$")

# Spoken punctuation, longest first so "dot com" is not eaten by "dot".
_SPOKEN_EMAIL = (
    (r"\bat\s+the\s+rate\s+(?:of\s+)?\b", "@"),
    (r"\b(?:at|attherate)\b", "@"),
    (r"\b(?:dot|full\s*stop|point)\b", "."),
    (r"\b(?:underscore|under\s*score)\b", "_"),
    (r"\b(?:hyphen|dash|minus)\b", "-"),
    (r"\bplus\b", "+"),
)

# Providers speech recognition reliably splits or mangles. Applied after the spoken
# punctuation pass, so "gee mail dot com" has already become "gee mail.com".
_DOMAIN_FIXES = (
    (r"\bg\s*(?:ee|e)?\s*mail\b", "gmail"),
    (r"\bgoogle\s*mail\b", "gmail"),
    (r"\b(?:yaho+|ya\s*hoo)\b", "yahoo"),
    (r"\bhot\s*mail\b", "hotmail"),
    (r"\bout\s*look\b", "outlook"),
    (r"\brediff\s*mail\b", "rediffmail"),
    (r"\bproton\s*mail\b", "protonmail"),
    (r"\bi\s*cloud\b", "icloud"),
)


# A model can emit a *syntactically valid* address that still has the spoken word baked
# in: "rahul_underscore_verma@outlook.com", "aditya.dot.p@gee.mail.com". Only fully
# delimited occurrences are replaced, so real names keep substrings like the "at" in
# "nathan" or the "dot" in "dotto".
_LEAKED_TOKENS = (
    (r"(?<=[._-])underscore(?=[._-])", "_"),
    (r"(?<=[._-])dot(?=[._-])", "."),
    (r"(?<=[._-])(?:dash|hyphen)(?=[._-])", "-"),
    (r"(?<=@)ge{1,2}\.?mail(?=\.)", "gmail"),
    (r"(?<=@)hot\.mail(?=\.)", "hotmail"),
    (r"(?<=@)out\.look(?=\.)", "outlook"),
    (r"(?<=@)i\.cloud(?=\.)", "icloud"),
)


def _repair_leaked_tokens(address: str) -> str:
    for pattern, symbol in _LEAKED_TOKENS:
        address = re.sub(pattern, symbol, address)
    return re.sub(r"([._-])\1+", r"\1", address)


def normalise_email(raw: str) -> str:
    """Turn a spoken email address into a real one.

    Every model tested garbles these differently - "rahul_underscore_verma@outlook.com",
    "aditya.p@ gmail.com", "aditya.dot.p@gee.mail.com" - and unlike the institution or
    PIN there is no directory to check the result against. So this is deliberately
    conservative: it only returns a rewrite that is a syntactically valid address, and
    otherwise hands back the original untouched rather than inventing one.
    """
    if not raw or not raw.strip():
        return raw
    text = raw.strip().lower()
    if EMAIL_RE.match(text):
        # Valid on its face, but may still spell out its own punctuation.
        repaired = _repair_leaked_tokens(text)
        return repaired if EMAIL_RE.match(repaired) else text

    text = re.sub(r"\s*@\s*", " at ", text)   # normalise so one code path handles both
    for pattern, symbol in _SPOKEN_EMAIL:
        text = re.sub(pattern, f" {symbol} ", text)
    for pattern, domain in _DOMAIN_FIXES:
        text = re.sub(pattern, domain, text)

    text = re.sub(r"\s*([@._+-])\s*", r"\1", text)   # drop spaces around punctuation
    text = re.sub(r"\s+", "", text)                  # "gee mail" -> already fixed above
    text = re.sub(r"\.{2,}", ".", text).strip(".")
    text = _repair_leaked_tokens(text)

    return text if EMAIL_RE.match(text) else raw



# ------------------------------------------------------- dates the candidate never gave

# Indian scripts write their own digits, and Whisper transcribes them as written.
_INDIC_DIGITS = str.maketrans("०१२३४५६७८९੦੧੨੩੪੫੬੭੮੯", "01234567890123456789")

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
          "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_WORD = re.compile(r"[a-z]+")


def _spoken_years(text: str) -> set[int]:
    """Years said in English words, which carry no digits for the literal scan to find.

    Covers the two ways anyone says a year aloud: "twenty twenty five" and "two thousand
    twenty five", plus "nineteen ninety nine". It has to exist - a candidate who said
    "graduated in twenty twenty five" would otherwise have their real year deleted as
    invented, which is exactly the silent data loss this whole layer is here to prevent.
    """
    words = _WORD.findall(text.lower())
    years: set[int] = set()
    for i, word in enumerate(words):
        if word not in _SMALL:
            continue
        head = _SMALL[word]
        rest = words[i + 1:i + 4]
        if head in (19, 20):                        # "nineteen ninety nine", "twenty five"
            tail = 0
            for part in rest[:2]:
                if part not in _SMALL:
                    break
                tail = tail + _SMALL[part] if tail else _SMALL[part]
            if 0 < tail < 100:
                years.add(head * 100 + tail)
        if head == 2 and rest[:1] == ["thousand"]:  # "two thousand twenty five"
            tail = 0
            for part in rest[1:3]:
                if part not in _SMALL:
                    break
                tail += _SMALL[part]
            years.add(2000 + tail)
    return years


def spoken_numbers(transcript: str) -> set[int]:
    """Every number the candidate can be said to have given, however they said it."""
    text = transcript.translate(_INDIC_DIGITS)
    numbers = {int(match) for match in re.findall(r"\d+", text)}
    numbers |= _spoken_years(text)
    for name, number in _MONTHS.items():            # "July" is as good as "07"
        if re.search(rf"\b{name}\b", text.lower()):
            numbers.add(number)
    return numbers


def _grounded_date(value: str, said: set[int]) -> bool:
    """Is every number in this date one the candidate actually gave?

    The extractor is told not to invent dates and does it anyway: "six months at a
    startup" came back as 2025-07-01 to 2025-12-31, and an ITI course with no year
    mentioned at all was dated 2026. Those numbers go on a document the candidate hands
    to an employer, so the rule is simply that each one has to have been said.
    """
    numbers = [int(n) for n in re.findall(r"\d+", value.translate(_INDIC_DIGITS))]
    # An academic year is written "2024-25", and neither the 25 nor the year 2025 was ever
    # said. Accept a two-digit part that closes a year the candidate gave, or the one
    # after it - which is what that notation means.
    short = {year % 100 for year in said if year > 1000}
    short |= {(year + 1) % 100 for year in said if year > 1000}
    return all(number in said or (number < 100 and number in short)
               for number in numbers)


# ------------------------------------------------- what a person is, not what they study

# A resume's headline is the job the person is applying for, and a trade is not a job
# title: a carpenter asked what work he wants says "carpentry", and the resume came back
# headed "Carpentry", which reads as a subject on a timetable rather than as a man looking
# for work. These are the trades this kiosk is for.
_TRADE_ROLES = {
    "carpentry": "Carpenter", "plumbing": "Plumber", "welding": "Welder",
    "masonry": "Mason", "painting": "Painter", "tailoring": "Tailor",
    "stitching": "Tailor", "driving": "Driver", "cooking": "Cook",
    "catering": "Cook", "baking": "Baker", "fitting": "Fitter",
    "turning": "Turner", "moulding": "Moulder", "machining": "Machinist",
    "wiring": "Electrician", "electrical": "Electrician",
    "electrical work": "Electrician", "electrician work": "Electrician",
    "plumber work": "Plumber", "mechanic work": "Mechanic",
    "motor mechanic": "Motor Mechanic", "beautician": "Beautician",
    "beauty culture": "Beautician", "hairdressing": "Hairdresser",
    "nursing": "Nurse", "teaching": "Teacher", "accounting": "Accountant",
    "accountancy": "Accountant", "security": "Security Guard",
    "housekeeping": "Housekeeper", "farming": "Farmer", "agriculture": "Farmer",
    "computer operating": "Computer Operator", "data entry": "Data Entry Operator",
    "refrigeration": "Refrigeration Technician", "ac repair": "AC Technician",
    "carpenter work": "Carpenter", "welding work": "Welder",
}


def normalise_role(raw: str) -> str:
    """The job title behind however the candidate described the work."""
    role = " ".join((raw or "").split())
    if not role:
        return role
    key = role.lower().strip(" .,-")   # a lookup key, so punctuation goes
    for filler in ("job of ", "work of ", "job", "ka kaam", "the "):
        key = key.removeprefix(filler).strip()
    return _TRADE_ROLES.get(key, role)


# Words that describe getting into a course rather than finishing one. An interview that
# produced "ITI Admission" as a degree is recording an enrolment, not a qualification.
_NOT_A_DEGREE = re.compile(
    r"\s*\b(admission|admissions|enrolment|enrollment|coaching|preparation)\b\s*",
    re.IGNORECASE)


def normalise_degree(raw: str) -> str:
    """Strip the words that describe enrolling rather than qualifying."""
    degree = _NOT_A_DEGREE.sub(" ", raw or "")
    # Trailing dots are left alone: "B.A." is spelt with one.
    return " ".join(degree.split()).strip(" ,-")

# ----------------------------------------------- the model apologising on the document

# A resume is not the place to explain what the interview did not get. Asked for the
# employer of a six-month internship the candidate never named, the extractor wrote
# "Startup (name not provided)" - and an employer reading the page sees the kiosk
# apologising in the middle of somebody's work history.
_PLACEHOLDER = re.compile(
    r"\s*[\(\[]\s*(?:name\s+)?(?:not\s+(?:provided|specified|mentioned|given|stated|"
    r"disclosed|available)|unnamed|unknown|unspecified|n/?a|tbd|none)\s*[\)\]]",
    re.IGNORECASE)
# The same thing said without brackets, where the whole value is the apology.
_ONLY_PLACEHOLDER = re.compile(
    r"^(?:not\s+(?:provided|specified|mentioned|given|stated)|unnamed|unknown|"
    r"unspecified|n/?a|none|tbd)\.?$", re.IGNORECASE)


def _without_placeholders(value: str) -> str:
    """Drop the model's notes about what it did not learn."""
    cleaned = _PLACEHOLDER.sub("", value or "").strip(" ,-")
    return "" if _ONLY_PLACEHOLDER.match(cleaned) else cleaned


def _scrub_placeholders(profile: dict, corrections: list) -> None:
    """Over every free-text field a resume actually prints."""
    targets = [(profile, key) for key in
               ("name", "target_role", "summary", "location", "email", "phone")]
    for entry in profile.get("education") or []:
        targets += [(entry, key) for key in ("degree", "institution", "location", "details")]
    for entry in profile.get("experience") or []:
        targets += [(entry, key) for key in ("title", "company")]
    for entry in profile.get("projects") or []:
        targets += [(entry, key) for key in ("name", "title", "description")]
    for holder, key in targets:
        value = holder.get(key)
        if not isinstance(value, str) or not value:
            continue
        cleaned = _without_placeholders(value)
        if cleaned != value:
            holder[key] = cleaned
            corrections.append({"field": key, "from": value, "to": cleaned,
                                "confidence": 1.0, "reason": "the model's own note"})


# ------------------------------------------------------------------ entry point

def apply_facts(profile: dict, transcript: str = "") -> dict:
    """Run every deterministic correction over an extracted profile.

    transcript is what the candidate actually said. Given it, any date in the profile
    built out of numbers nobody spoke is removed - see _grounded_date. Without it that
    check is skipped, because a date cannot be contradicted by evidence that is not there.

    Adds a "_corrections" list describing what changed, so the UI can show the
    candidate what was auto-fixed instead of silently rewriting their answers.
    """
    corrections = []
    validation_warnings = list(profile.get("_validation_warnings", []))
    _scrub_placeholders(profile, corrections)

    role = profile.get("target_role", "")
    fixed_role = normalise_role(role)
    if fixed_role != role:
        profile["target_role"] = fixed_role
        corrections.append({"field": "target_role", "from": role, "to": fixed_role,
                            "confidence": 1.0})

    for edu in profile.get("education", []):
        degree = edu.get("degree", "")
        fixed_degree = normalise_degree(degree)
        if fixed_degree != degree:
            edu["degree"] = fixed_degree
            corrections.append({"field": "education.degree", "from": degree,
                                "to": fixed_degree, "confidence": 1.0})
        original = edu.get("institution", "")
        fixed, score = correct_institution(original)
        if fixed != original:
            edu["institution"] = fixed
            corrections.append({
                "field": "education.institution",
                "from": original, "to": fixed, "confidence": round(score, 2),
            })

    if transcript:
        said = spoken_numbers(transcript)
        dated = [(entry, field) for entry in profile.get("education") or []
                 for field in ("year",)]
        dated += [(entry, field) for entry in profile.get("experience") or []
                  for field in ("start", "end")]
        for entry, field in dated:
            value = (entry.get(field) or "").strip()
            if value and not _grounded_date(value, said):
                entry[field] = ""
                corrections.append({"field": f"date.{field}", "from": value, "to": "",
                                    "confidence": 1.0, "reason": "never said"})

    # An internship is very often at a college, and the garbled name lands in "company"
    # where nothing was looking at it: one resume carried "Chipkare University" as the
    # employer while the education section had been tidied up. Only names that call
    # themselves an institution are matched, so ordinary employers are never dragged
    # towards a university that happens to share a word with them.
    for job in profile.get("experience", []):
        company = (job.get("company") or "").strip()
        if not company or not _generic_word(company):
            continue
        fixed, score = correct_institution(company)
        if fixed != company:
            job["company"] = fixed
            corrections.append({"field": "experience.company", "from": company,
                                "to": fixed, "confidence": round(score, 2)})

    phone = profile.get("phone", "")
    fixed_phone = normalise_phone(phone)
    if fixed_phone != phone:
        profile["phone"] = fixed_phone
        corrections.append({"field": "phone", "from": phone, "to": fixed_phone,
                            "confidence": 1.0})

    email = profile.get("email", "")
    fixed_email = normalise_email(email)
    if fixed_email != email:
        profile["email"] = fixed_email
        corrections.append({"field": "email", "from": email, "to": fixed_email,
                            "confidence": 1.0})
    elif email.strip() and not EMAIL_RE.match(email.strip().lower()):
        # Could not be repaired into a valid address. Say so rather than printing a
        # broken email: there is no directory to verify a personal address against.
        warning = {"field": "email", "code": "email_not_valid", "value": email}
        if warning not in validation_warnings:
            validation_warnings.append(warning)

    # "Rajpura, PIN 140401" is how it comes back when the candidate says the words out
    # loud. The label is not part of an address and does not belong on a resume.
    location = _LABELLED_PIN.sub("", profile.get("location", "")).strip(" ,")
    if location != profile.get("location", ""):
        corrections.append({"field": "location", "from": profile["location"],
                            "to": location, "confidence": 1.0})
        profile["location"] = location
    pin = lookup_pincode(location)
    if pin and pin["verified"] and pin["place_match"] is False:
        # The PIN itself is valid, but none of its post offices/district/state names
        # matched the supplied place. Do not silently combine conflicting locations.
        warning = {
            "field": "location",
            "code": "pincode_place_not_matched",
            "value": location,
            "pincode": pin["pincode"],
            "expected_districts": pin["districts"],
            "expected_state": pin["state"],
            "source": pin["source"],
        }
        if warning not in validation_warnings:
            validation_warnings.append(warning)
    elif pin and pin["verified"]:
        additions = []
        location_norm = _normalise_place(location, pin["pincode"])

        # Add a district only when a local office/district/division matched. A PIN
        # with no place text, or only a state match, is not enough to choose one.
        district = _strip_postal_suffix(pin.get("district") or "") or None
        if (
            district
            and pin.get("match_field") in {"officename", "district", "divisionname"}
            and _place_score(location_norm, _normalise_place(district)) < 0.86
        ):
            additions.append(district)

        state = pin.get("state")
        if state and _place_score(location_norm, _normalise_place(state)) < 0.86:
            additions.append(state)

        if additions:
            profile["location"] = ", ".join([location.strip(), *additions]).strip(", ")
            corrections.append({
                "field": "location", "from": location, "to": profile["location"],
                "confidence": pin["confidence"],
                "source": pin["source"],
            })
    elif pin is None and PIN_RE.search(location):
        # lookup_pincode returns None only when the directory answered and had no such
        # PIN - being offline gives a result with verified=False instead. So this is a
        # six-digit number that is not an Indian PIN code, and it was going onto the
        # resume as the candidate's address without a word: one interview printed
        # "Rajpura, 166001", a town whose real PIN is 140401.
        warning = {
            "field": "location",
            "code": "pincode_not_found",
            "value": location,
            "pincode": PIN_RE.search(location).group(1),
        }
        if warning not in validation_warnings:
            validation_warnings.append(warning)

    if corrections:
        profile["_corrections"] = corrections
    if validation_warnings:
        profile["_validation_warnings"] = validation_warnings
    return profile
